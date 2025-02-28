# main.py

from advice import blackjack_advice

def get_card_input(prompt):
    """
    Helper function to get card input from the user.
    Converts input to integers and handles Aces (11).
    """
    while True:
        try:
            cards = input(prompt).strip().split()
            cards = [int(card) if card != 'A' else 11 for card in cards]
            return cards
        except ValueError:
            print("Invalid input. Please enter numbers or 'A' for Ace.")

def get_dealer_card_input(prompt):
    """
    Helper function to get the dealer's card input from the user.
    Converts input to an integer and handles Aces (11).
    """
    while True:
        try:
            card = input(prompt).strip()
            if card == 'A':
                return 11
            return int(card)
        except ValueError:
            print("Invalid input. Please enter a number or 'A' for Ace.")

def main():
    print("Welcome to the Blackjack Advice Tool!")
    
    while True:
        print("\n--- New Hand ---")
        
        # Get player's hand
        player_hand = get_card_input("Enter the player's cards (e.g., '10 7' or 'A 5'): ")
        
        # Get dealer's card
        dealer_card = get_dealer_card_input("Enter the dealer's visible card (e.g., '6' or 'A'): ")
        
        # Get advice
        advice = blackjack_advice(player_hand, dealer_card)
        
        # Display results
        print("\n--- Results ---")
        print(f"Player's Hand: {player_hand}")
        print(f"Dealer's Card: {dealer_card}")
        print(f"Advice: {advice}")
        
        # Ask if the user wants to continue
        again = input("\nDo you want to test another hand? (y/n): ").strip().lower()
        if again != 'y':
            print("Thanks for using the Blackjack Advice Tool! Goodbye!")
            break

if __name__ == "__main__":
    main()