#game.py
from copy import deepcopy  # To create deep copies of complex data structures

round_number = round

class G08A():
    def __init__(self, players, teams) -> None:
        """
        Initialization now includes teams.
        
        Changes:
        - Added 'teams' as a parameter and stored it as self.teams.
        
        Purpose:
        The game now involves two teams of players competing rather than individuals.
        Each team is represented as a list of player names. This allows us to determine
        winners based on team averages rather than the closest individual player.
        """
        self.all_players = players[::]
        self.survival_players = players[::]
        self.round_winner = {}
        self.teams = teams  # A list of lists, each sub-list is a team of player names.
        self.full_round_history = []
    
    def daily_bidding(self, players):
        Average = 0
        for player in players:
            player.act()
            Average += player.last_bidding
        Average /= len(players)
        Target = round_number(Average * 0.8, 2)
        return Average, Target
    
    def round_deduction(self, players, winner):
        """
        Deduct HP from players not in the winning team.
        
        No direct changes needed here because:
        - Previously, it checked if a player's name is not in 'winner' (a list of winners).
        - Now 'winner' is a list of player names from the winning team.
        
        This logic is still compatible with the team-based approach. If a team is the winner,
        all its members will be in 'winner', and all others will lose 1 HP.
        """
        for player in players:
            if player.name not in winner:
                player.deduction(1)
    
    def check_winner(self, players, target):
        """
        Determine the winning team based on which team's average bid is closest to the target.
        
        Changes:
        - Instead of identifying a single player's closest bid, we now:
          1. Compute each team's average bidding.
          2. Find the team whose average is closest to the target.
          3. If one team is closer, return that team's player names as winners.
          4. If it's a tie, return an empty list.
        
        Purpose:
        This modification aligns the winner determination step with the new team-based mechanics.
        """
        team_results = []
        for team in self.teams:
            # Filter players belonging to this team
            team_players = [p for p in players if p.name in team]
            if not team_players:
                continue  # Skip if no players are left from this team
            
            # Calculate team's average bid
            team_avg = sum(p.last_bidding for p in team_players) / len(team_players)
            diff = abs(team_avg - target)
            team_results.append((diff, team, team_avg))
        
        # If no teams or just one surviving team
        if len(team_results) == 0:
            return [], None
        if len(team_results) == 1:
            # Only one team alive, automatically wins
            return team_results[0][1], team_results[0][2]
        
        # Sort teams by their closeness to target
        team_results.sort(key=lambda x: x[0])
        # Check if there's a tie
        if len(team_results) > 1 and team_results[0][0] == team_results[1][0]:
            # Tie scenario: return empty list
            return [], None
        else:
            # Clear winner
            winner_team = team_results[0][1]
            winner_avg = team_results[0][2]
            return winner_team, winner_avg
    
    def check_tie(self, players):
        # Determine which teams are alive
        team_averages = []
        for team in self.teams:
            # Get players from this team who are still alive
            team_players = [p for p in players if p.name in team]

            if team_players:
                # Since the entire team is alive or dead as a unit, we can safely compute the average
                avg_bid = sum(p.last_bidding for p in team_players) / len(team_players)
                team_averages.append(avg_bid)

        # If less than two teams are alive, can't have a tie
        if len(team_averages) < 2:
            return False

        # Check if all alive teams have the same average
        return len(set(team_averages)) == 1
        
    def run_single_round(self, round_id):
        """
        Execute a single round of the game with team-based winner determination.
        
        Changes:
        - Incorporate team-based winner logic in place of individual winner logic.
        - Update the messaging to refer to teams instead of individual winners.
        """
        for player in self.survival_players:
            player.start_round(round_id, self.full_round_history)
        
        Average, Target = self.daily_bidding(self.survival_players)
        round_history_entry = {
        "round_id": round_id,
        "bids": {p.name: p.last_bidding for p in self.survival_players},
        "target": Target
        }
        self.full_round_history.append(round_history_entry)

        Tie_status = self.check_tie(self.survival_players)
        
        if Tie_status:
            # All players chose the same number; no team wins
            WINNER = []
            WINNER_BID = None
        else:
            WINNER, WINNER_BID = self.check_winner(self.survival_players, Target)
        
        self.round_winner[round_id] = WINNER
        self.round_deduction(self.survival_players, WINNER)
        
        bidding_numbers = [f"{player.last_bidding}" for player in self.survival_players]
        history_biddings = {player.name: deepcopy(player.biddings) for player in self.survival_players}
        bidding_details_list = [f"{player.name} chose {player.last_bidding}" for player in self.survival_players]
        diff_details_list = [
            f"{player.name}: |{player.last_bidding} - {Target}| = {round_number(abs(player.last_bidding - Target))}"
            for player in self.survival_players
        ]
        player_details_list = [player.show_info() for player in self.survival_players]
        
        bidding_numbers_str = " + ".join(bidding_numbers)
        bidding_details_str = ", ".join(bidding_details_list)
        diff_details_str = ", ".join(diff_details_list)
        player_details_str = ", ".join(player_details_list)
        
        # Prepare round result messages
        if Tie_status:
            # Universal tie: everyone loses 1 point
            BIDDING_INFO = (
                f"Thank you all for participating in Round {round_id}. In this round, {bidding_details_str}.\n"
                "All players chose the same number, resulting in a tie. All players lose 1 point. "
                f"After the deduction, player information is: {player_details_str}."
            )
        else:
            if WINNER and WINNER_BID is not None:
                # WINNER is a list of player names in the winning team
                winner_team_str = ", ".join(WINNER)
                BIDDING_INFO = (
                    f"Thank you all for participating in Round {round_id}. In this round, {bidding_details_str}.\n"
                    f"The average is ({bidding_numbers_str}) / {len(self.survival_players)} = {Average}.\n"
                    f"The average {Average} multiplied by 0.8 equals {Target}.\n"
                    f"{diff_details_str}\n"
                    f"The winning team's average choice of {WINNER_BID} is closest to {Target}. "
                    f"Team members {winner_team_str} win this round. All other players lose 1 point. "
                    f"After the deduction, player information is: {player_details_str}."
                )
            else:
                # It's a team-level tie: teams equally close to target
                BIDDING_INFO = (
                    f"Thank you all for participating in Round {round_id}. In this round, {bidding_details_str}.\n"
                    f"The average is ({bidding_numbers_str}) / {len(self.survival_players)} = {Average}.\n"
                    f"The average {Average} multiplied by 0.8 equals {Target}.\n"
                    f"{diff_details_str}\n"
                    "It's a tie between the teams! All players lose 1 point. "
                    f"After the deduction, player information is: {player_details_str}."
                )
        
        # Update players about the round results and handle eliminations
        survival_players = []
        dead_players = []
        for player in self.survival_players:
            win = player.name in WINNER
            player.notice_round_result(round_id, BIDDING_INFO, Target, win, bidding_details_str, history_biddings)
            
            if player.hp <= 0:
                dead_players.append(player)
            else:
                survival_players.append(player)
        
        self.survival_players = survival_players
        
        # Notify surviving agent players about eliminated players
        for out in dead_players:
            for other_player in survival_players:
                if other_player.is_agent:
                    other_player.message += [
                        {"role": "system", "content": f"{out.name}'s hp is below 0, so {out.name} has been eliminated from the challenge!"}
                    ]
        
        for player in self.survival_players:
            player.end_round()
        
        print("Round ", round_id, ": ", bidding_details_str)
    
    def run_multi_round(self, max_round):
        """
        Runs multiple rounds until 'max_round' is reached.
        """
        for player in self.all_players:
            player.ROUND_WINNER = self.round_winner
        
        for i in range(1, max_round+1):
            self.run_single_round(i)
