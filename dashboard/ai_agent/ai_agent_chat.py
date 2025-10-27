# ai_agent_chat.py
"""
AI AGENT CHAT INTERFACE
-----------------------
Simple command-line chat interface for the AI Agent.

Usage: python ai_agent_chat.py
"""

from ai_agent_parser import QueryParser
from ai_agent_responder import ResponseGenerator

def main():
    """Run the AI Agent chat interface."""
    
    print("\n" + "="*70)
    print(" ENRON RISK AI AGENT")
    print("="*70)
    print("\nHello! I can answer questions about employee risk.")
    print("\nExample questions:")
    print("  • Who is high risk this week?")
    print("  • Why is christy.wire@enron.com high risk?")
    print("  • How many people are critical?")
    print("  • Show trend for mark.fisher@enron.com")
    print("\nType 'quit' to exit")
    print("="*70 + "\n")
    
    # Initialize components
    parser = QueryParser()
    responder = ResponseGenerator()
    
    # Chat loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
            print("\n Goodbye! Stay safe!\n")
            break
        
        # Skip empty input
        if not user_input:
            continue
        
        # Process query
        try:
            # Parse query
            parsed = parser.parse(user_input)
            
            # Generate response
            response = responder.generate_response(parsed)
            
            # Display response
            print(f"\n AI Agent:\n{response}\n")
            print("-"*70 + "\n")
            
        except Exception as e:
            print(f"\n Error: {e}\n")
            print("Please try rephrasing your question.\n")


if __name__ == "__main__":
    main()