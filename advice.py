# advice.py

def blackjack_advice(player_hand, dealer_card):
    """
    Provides Blackjack advice based on the player's hand and the dealer's visible card.
    
    Args:
        player_hand (list): List of card values in the player's hand (e.g., [10, 7]).
        dealer_card (int): Value of the dealer's visible card (e.g., 6).
    
    Returns:
        str: Advice ("Hit", "Stand", "Double", "Split").
    """
    
    # Calculate the sum of the player's hand
    hand_sum = sum(player_hand)
    
    # Check if the hand is soft (contains an Ace counted as 11)
    is_soft = 11 in player_hand and hand_sum <= 21
    
    # Adjust for soft hands
    if is_soft:
        hand_sum -= 10  # Treat Ace as 1 temporarily
    
    # Basic strategy rules
    if len(player_hand) == 2 and player_hand[0] == player_hand[1]:
        # Pair splitting logic
        if player_hand[0] in [8, 11]:  # Always split 8s and Aces
            return "Split"
        elif player_hand[0] in [2, 3, 7] and dealer_card <= 7:
            return "Split"
        elif player_hand[0] == 6 and dealer_card <= 6:
            return "Split"
        elif player_hand[0] == 9 and dealer_card not in [7, 10, 11]:
            return "Split"
        elif player_hand[0] == 4 and dealer_card in [5, 6]:
            return "Split"
    
    if is_soft:
        # Soft hand logic
        if hand_sum <= 17:
            return "Hit"
        elif hand_sum == 18 and dealer_card in [9, 10, 11]:
            return "Hit"
        else:
            return "Stand"
    else:
        # Hard hand logic
        if hand_sum <= 11:
            return "Hit"
        elif hand_sum == 12 and dealer_card in [4, 5, 6]:
            return "Stand"
        elif 13 <= hand_sum <= 16 and dealer_card >= 7:
            return "Hit"
        else:
            return "Stand"