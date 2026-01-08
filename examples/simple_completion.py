"""
Simple completion example for fastlitellm.

This example demonstrates basic chat completion usage.
"""

import os
import fastlitellm

# Ensure API key is set
# export OPENAI_API_KEY="your-key"


def main():
    """Run a simple completion."""
    # Simple completion
    response = fastlitellm.completion(
        model="gpt-4o-mini",  # or "openai/gpt-4o-mini"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    # Access the response
    print("Response ID:", response.id)
    print("Model:", response.model)
    print("Content:", response.choices[0].message.content)
    print("Finish reason:", response.choices[0].finish_reason)

    # Access usage
    if response.usage:
        print("\nUsage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")

    # Calculate cost
    try:
        cost = fastlitellm.completion_cost(response)
        print(f"\nEstimated cost: ${cost:.6f}")
    except Exception as e:
        print(f"\nCould not calculate cost: {e}")


def example_with_different_providers():
    """Examples with different providers."""
    # OpenAI
    # response = fastlitellm.completion(
    #     model="openai/gpt-4o-mini",
    #     messages=[{"role": "user", "content": "Hello!"}]
    # )

    # Anthropic (set ANTHROPIC_API_KEY)
    # response = fastlitellm.completion(
    #     model="anthropic/claude-3-5-sonnet-latest",
    #     messages=[{"role": "user", "content": "Hello!"}]
    # )

    # Gemini (set GEMINI_API_KEY or GOOGLE_API_KEY)
    # response = fastlitellm.completion(
    #     model="gemini/gemini-1.5-flash",
    #     messages=[{"role": "user", "content": "Hello!"}]
    # )

    # Groq (set GROQ_API_KEY)
    # response = fastlitellm.completion(
    #     model="groq/llama-3.1-70b-versatile",
    #     messages=[{"role": "user", "content": "Hello!"}]
    # )
    pass


if __name__ == "__main__":
    main()
