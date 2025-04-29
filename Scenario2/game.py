from copy import deepcopy  # To create deep copies of complex data structures

round_number = round

class G08A():
    def __init__(self, players, teams) -> None:
        self.all_players = players[::]
        self.survival_players = players[::]
        self.round_winner = {}
        self.teams = teams
        self.full_round_history = []  # Store global bidding history

    def collaborative_bidding(self, players, round_id):
        team_index_by_player = {}
        for idx, team in enumerate(self.teams):
            for name in team:
                team_index_by_player[name] = idx

        team_players = {}
        for idx, team in enumerate(self.teams):
            team_players[idx] = [p for p in players if p.name in team]

        for p in players:
            p.act_initial(round_id, self.full_round_history)

        initial_proposals = {}
        for idx, t_players in team_players.items():
            initial_proposals[idx] = {}
            for p in t_players:
                initial_proposals[idx][p.name] = p.initial_bid

        all_feedback = {}
        for idx, t_players in team_players.items():
            all_feedback[idx] = {}
            for p in t_players:
                proposals_excluding_self = []
                for other_name, bid in initial_proposals[idx].items():
                    if other_name != p.name:
                        proposals_excluding_self.append(f"{other_name} proposed {bid}")
                proposals_summary = "\n".join(proposals_excluding_self)

                fb_text = p.act_feedback(round_id, proposals_summary)
                all_feedback[idx][p.name] = fb_text

        for idx, t_players in team_players.items():
            for p in t_players:
                summary_of_feedback_for_p = ""
                for other_player_name, fb_text in all_feedback[idx].items():
                    if other_player_name != p.name:
                        summary_of_feedback_for_p += (
                            f"{other_player_name}'s feedback on your proposal:\n{fb_text}\n\n"
                        )
                if summary_of_feedback_for_p.strip():
                    p.receive_teammates_feedback(round_id, summary_of_feedback_for_p)

        for p in players:
            p.act_final(round_id)

        final_bids = [p.last_bidding for p in players]
        Average = sum(final_bids) / len(final_bids)
        Target = round_number(Average * 0.8, 2)

        return Average, Target

    def round_deduction(self, players, winner):
        for player in players:
            if player.name not in winner:
                player.deduction(1)

    def check_winner(self, players, target):
        team_results = []
        for team in self.teams:
            team_players = [p for p in players if p.name in team]
            if not team_players:
                continue
            team_avg = sum(p.last_bidding for p in team_players) / len(team_players)
            diff = abs(team_avg - target)
            team_results.append((diff, team, team_avg))

        if len(team_results) == 0:
            return [], None
        if len(team_results) == 1:
            return team_results[0][1], team_results[0][2]

        team_results.sort(key=lambda x: x[0])
        if len(team_results) > 1 and team_results[0][0] == team_results[1][0]:
            return [], None
        else:
            winner_team = team_results[0][1]
            winner_avg = team_results[0][2]
            return winner_team, winner_avg

    def check_tie(self, players):
        team_averages = []
        for team in self.teams:
            team_players = [p for p in players if p.name in team]
            if team_players:
                avg_bid = sum(p.last_bidding for p in team_players) / len(team_players)
                team_averages.append(avg_bid)

        if len(team_averages) < 2:
            return False
        return len(set(team_averages)) == 1

    def run_single_round(self, round_id):
        for player in self.survival_players:
            player.start_round(round_id)

        Average, Target = self.collaborative_bidding(self.survival_players, round_id)

        Tie_status = self.check_tie(self.survival_players)
        if Tie_status:
            WINNER = []
            WINNER_BID = None
        else:
            WINNER, WINNER_BID = self.check_winner(self.survival_players, Target)

        self.round_winner[round_id] = WINNER
        self.round_deduction(self.survival_players, WINNER)

        bidding_numbers = [str(p.last_bidding) for p in self.survival_players]
        history_biddings = {p.name: deepcopy(p.biddings) for p in self.survival_players}
        bidding_details_list = [f"{p.name} chose {p.last_bidding}" for p in self.survival_players]
        diff_details_list = [
            f"{p.name}: |{p.last_bidding} - {Target}| = {round_number(abs(p.last_bidding - Target))}"
            for p in self.survival_players
        ]
        player_details_list = [p.show_info() for p in self.survival_players]

        bidding_numbers_str = " + ".join(bidding_numbers)
        bidding_details_str = ", ".join(bidding_details_list)
        diff_details_str = ", ".join(diff_details_list)
        player_details_str = ", ".join(player_details_list)

        round_history_entry = {
            "round_id": round_id,
            "bids": {p.name: p.last_bidding for p in self.survival_players},
            "target": Target
        }
        self.full_round_history.append(round_history_entry)

        if Tie_status:
            BIDDING_INFO = (
                f"Thank you all for participating in Round {round_id}. "
                f"In this round, {bidding_details_str}.\n"
                f"It's a tie for the closest team average to the target {Target}! All players lose 1 point. "
                f"After the deduction, player information is: {player_details_str}."
            )
        else:
            if WINNER and WINNER_BID is not None:
                winner_team_str = ", ".join(WINNER)
                BIDDING_INFO = (
                    f"Thank you all for participating in Round {round_id}. "
                    f"In this round, {bidding_details_str}.\n"
                    f"The average is ({bidding_numbers_str}) / {len(self.survival_players)} = {Average}.\n"
                    f"The average {Average} multiplied by 0.8 equals {Target}.\n"
                    f"{diff_details_str}\n"
                    f"The winning team's average choice of {WINNER_BID} is closest to {Target}. "
                    f"Team members {winner_team_str} win this round. All other players lose 1 point. "
                    f"After the deduction, player information is: {player_details_str}."
                )
            else:
                BIDDING_INFO = (
                    f"Thank you all for participating in Round {round_id}. "
                    f"In this round, {bidding_details_str}.\n"
                    f"The average is ({bidding_numbers_str}) / {len(self.survival_players)} = {Average}.\n"
                    f"The average {Average} multiplied by 0.8 equals {Target}.\n"
                    f"{diff_details_str}\n"
                    "It's a tie between the teams! All players lose 1 point. "
                    f"After the deduction, player information is: {player_details_str}."
                )

        survival_players = []
        dead_players = []
        for p in self.survival_players:
            win = p.name in WINNER
            p.notice_round_result(round_id, BIDDING_INFO, Target, win, bidding_details_str, history_biddings)
            if p.hp <= 0:
                dead_players.append(p)
            else:
                survival_players.append(p)

        self.survival_players = survival_players

        for out in dead_players:
            for other_player in survival_players:
                if other_player.is_agent:
                    other_player.message.append(
                        {
                            "role": "system",
                            "content": f"{out.name}'s hp is below 0, so {out.name} has been eliminated from the challenge!"
                        }
                    )

        for player in self.survival_players:
            player.end_round()

        print(f"Round {round_id}: {bidding_details_str}")

    def run_multi_round(self, max_round):
        for player in self.all_players:
            player.ROUND_WINNER = self.round_winner

        for i in range(1, max_round+1):
            self.run_single_round(i)
