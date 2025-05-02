# A Study in Scarlet and Silicon: Co-Creation in a Murder Mystery

This project introduces **Baker Street Triad**, a Sherlock Holmes-inspired text-based murder mystery game powered by dual Large Language Models (LLMs). Unlike traditional LLM-driven games with an all-knowing narrator, our system promotes **co-creative storytelling** by mimicking the Holmes-Watson partnership dynamic.

## 🔍 Concept

* **Player as Sherlock Holmes** investigates mysteries.
* **Watson (LLM)** is a companion with limited knowledge—just like the player.
* **Game Master (LLM)** knows the full narrative and drives the story.

This setup fosters collaboration, not command-following. Watson offers insights based only on known clues, preserving the intrigue and supporting player-driven deduction.

## 🧠 Key Features

* **Dual LLM Architecture**:

  * *Llama 3.2*: Omniscient Game Master for complex, canonical storytelling.
  * *TinyLlama*: Fast, non-omniscient Watson for responsive dialogue.

* **RAG (Retrieval-Augmented Generation)**:

  * Integrates the full Sherlock Holmes corpus and chat history for grounded, immersive interactions.

* **Offline Capability**: No cloud dependency; fully local execution.

## 🧪 User Study Highlights

* High engagement and enjoyment (9.25/10 overall score).
* Strong collaboration with Watson (5/5 teamwork rating).
* Suggestions for improvement included:

  * More visual content.
  * Making Watson more proactive.