"""Interactive local RAG assistant for restaurant reviews (Ollama + ChromaDB + LangChain)."""

from __future__ import annotations

import argparse
from collections import deque

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

import config
from retriever import HybridRetriever, detect_rating_filter, detect_restaurant
from vector import build_vector_store, compute_stats

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant that answers questions about restaurants using only the
evidence below: review excerpts and pre-computed statistics.

Rules:
- Ground every claim in the numbered reviews and cite them like [R1], [R3] (statistics need no citation).
- Use the statistics for questions about counts, averages or rating distribution.
- Reviews belong to the restaurant named in brackets; never mix restaurants up.
- If the evidence is insufficient, say so plainly. Never invent menu items,
  prices, policies or experiences.
- State whether the evidence is positive, negative or mixed, and mention how many
  reviews support a point when that is useful.
- Keep the answer concise and well organised. Use short bullet points for lists.

Statistics:
{stats}

Reviews:
{reviews}

Conversation so far:
{history}

Question: {question}

Answer:"""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's latest question as a single standalone search query, using the
conversation only to resolve pronouns and omitted context (for example restaurant names).
Do not add topics, restaurants or details the latest question does not refer to.
If the latest question is already self-contained, return it unchanged.
Return only the rewritten question, with no explanation.

Conversation:
{history}

Latest question: {question}

Standalone question:"""
)

HELP = """Commands:
  /restaurants        list indexed restaurants
  /use <name>|all     restrict answers to one restaurant (or clear the filter)
  /stats              show review counts and average ratings
  /sources on|off     show the retrieved reviews under each answer
  /reindex            rebuild the whole index from the CSV files
  /clear              forget the conversation history
  /help               show this help
  q                   quit"""


class Assistant:
    def __init__(self, show_sources: bool, restaurant: str | None, force_rebuild: bool):
        print("Preparing index (first run embeds every review and may take a while)...")
        self.store = build_vector_store(force_rebuild=force_rebuild)
        self.retriever = HybridRetriever(self.store)
        self.stats = compute_stats()
        self.show_sources = show_sources
        self.scope: str | None = None
        self.history: deque[tuple[str, str]] = deque(maxlen=config.HISTORY_TURNS)

        def make_llm(temperature: float) -> OllamaLLM:
            return OllamaLLM(
                model=config.LLM_MODEL,
                temperature=temperature,
                num_ctx=config.LLM_NUM_CTX,
                base_url=config.OLLAMA_BASE_URL,
            )

        self.answer_chain = ANSWER_PROMPT | make_llm(config.LLM_TEMPERATURE) | StrOutputParser()
        self.rewrite_chain = REWRITE_PROMPT | make_llm(0.0) | StrOutputParser()
        if restaurant:
            self.set_scope(restaurant)

    # -- commands ----------------------------------------------------------
    def resolve_restaurant(self, text: str) -> str | None:
        names = self.retriever.restaurants
        for name in names:
            if name.lower() == text.lower():
                return name
        found = [n for n in names if text.lower() in n.lower()]
        return found[0] if len(found) == 1 else None

    def set_scope(self, text: str) -> None:
        if text.lower() == "all":
            self.scope = None
            print("Scope: all restaurants.\n")
            return
        name = self.resolve_restaurant(text)
        if name is None:
            print(f"No unique restaurant matches '{text}'. Try /restaurants.\n")
            return
        self.scope = name
        print(f"Scope: {name}\n")

    def command(self, line: str) -> None:
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd == "/help":
            print(HELP + "\n")
        elif cmd == "/restaurants":
            for name in self.retriever.restaurants:
                print(f"  - {name}")
            print()
        elif cmd == "/use":
            self.set_scope(arg or "all")
        elif cmd == "/stats":
            for stat in self.stats.values():
                print(f"  {stat.summary()}")
            print()
        elif cmd == "/sources":
            self.show_sources = arg.lower() != "off"
            print(f"Sources {'on' if self.show_sources else 'off'}.\n")
        elif cmd == "/clear":
            self.history.clear()
            print("History cleared.\n")
        elif cmd == "/reindex":
            self.store = build_vector_store(force_rebuild=True)
            self.retriever = HybridRetriever(self.store)
            self.stats = compute_stats()
            print("Index rebuilt.\n")
        else:
            print("Unknown command. Type /help.\n")

    # -- question answering -----------------------------------------------
    def format_history(self) -> str:
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.history) or "(none)"

    def standalone_question(self, question: str) -> str:
        if not self.history:
            return question
        rewritten = self.rewrite_chain.invoke(
            {"history": self.format_history(), "question": question}
        ).strip()
        return rewritten.splitlines()[0] if rewritten else question

    def ask(self, question: str) -> None:
        search_query = self.standalone_question(question)
        restaurant = self.scope or detect_restaurant(search_query, self.retriever.restaurants)
        low, high = detect_rating_filter(search_query)

        docs = self.retriever.retrieve(search_query, restaurant, low, high)
        if (low is not None or high is not None) and len(docs) < 3:
            docs = self.retriever.retrieve(search_query, restaurant)  # filter too strict
        if not docs:
            print("\nAssistant: I couldn't find relevant reviews for that question.\n")
            return

        reviews = "\n\n".join(
            f"[R{i}] ({d.metadata.get('restaurant')}, rating {d.metadata.get('rating')}/5, "
            f"{d.metadata.get('date')}) {d.page_content}"
            for i, d in enumerate(docs, start=1)
        )
        relevant = [restaurant] if restaurant else list(self.stats)  # no single restaurant: give all stats
        stats = "\n".join(self.stats[r].summary() for r in relevant if r in self.stats)

        if search_query != question:
            print(f"(searching for: {search_query})")
        print("\nAssistant: ", end="", flush=True)
        parts: list[str] = []
        for token in self.answer_chain.stream(
            {
                "stats": stats or "(none)",
                "reviews": reviews,
                "history": self.format_history(),
                "question": question,
            }
        ):
            parts.append(token)
            print(token, end="", flush=True)
        print("\n")
        self.history.append((question, "".join(parts).strip()))

        if self.show_sources:
            print("Sources:")
            for i, d in enumerate(docs, start=1):
                text = d.page_content.split("] ", 1)[-1]
                print(f"  [R{i}] {d.metadata.get('restaurant')} | {d.metadata.get('rating')}/5 | {text[:110]}")
            print()

    def run(self) -> None:
        names = ", ".join(self.retriever.restaurants)
        print(f"\nLocal Restaurant Review Assistant  ({config.LLM_MODEL})")
        print(f"Indexed restaurants: {names}")
        print("Ask a question, or type /help. Type 'q' to exit.\n")

        while True:
            try:
                prompt = f"You [{self.scope}]: " if self.scope else "You: "
                line = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return
            if line.lower() in {"q", "quit", "exit"}:
                print("Goodbye!")
                return
            if not line:
                continue
            try:
                if line.startswith("/"):
                    self.command(line)
                else:
                    self.ask(line)
            except Exception as exc:  # Surface actionable local setup errors.
                print(f"\nError: {exc}")
                print(
                    "Check that Ollama is running and that "
                    f"'{config.LLM_MODEL}' and '{config.EMBEDDING_MODEL}' are installed.\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restaurant", help="start scoped to one restaurant (name or part of it)")
    parser.add_argument("--sources", action="store_true", help="show retrieved reviews under answers")
    parser.add_argument("--reindex", action="store_true", help="rebuild the vector index first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        Assistant(args.sources, args.restaurant, args.reindex).run()
    except Exception as exc:
        raise SystemExit(
            f"Startup failed: {exc}\n"
            "Make sure Ollama is running and the models are pulled (see README)."
        )


if __name__ == "__main__":
    main()
