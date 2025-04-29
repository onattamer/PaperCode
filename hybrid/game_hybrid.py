from copy import deepcopy  # For copying complex data

round_number = round

class G08AHybrid:
    """
    Hybrid version of the Guess 0.8 of the Average game.
    Allows one team to use collaborative (Scenario 2) agents and
    the other team to use solo (Scenario 1) agents.
    """
    def __init__(self, players, teams) -> None:
        self.all_players = players[::]
        self.survival_players = players[::]
        self.teams = teams
        self.round_winner = {}
        self.full_round_history = []

    def is_collaborative(self, player):
        """Detect if a player supports the collaborative API (Scenario 2)."""
        return hasattr(player, 'act_initial') and hasattr(player, 'act_feedback') and hasattr(player, 'act_final')

    def round_deduction(self, players, winner):
        """Deduct HP from players not in the winning team."""
        for p in players:
            if p.name not in winner:
                p.deduction(1)

    def check_winner(self, players, target):
        """Determine the winning team by closest team-average to target."""
        team_results = []
        for team in self.teams:
            team_players = [p for p in players if p.name in team]
            if not team_players:
                continue
            team_avg = sum(p.last_bidding for p in team_players) / len(team_players)
            diff = abs(team_avg - target)
            team_results.append((diff, team, team_avg))
        if not team_results:
            return [], None
        if len(team_results) == 1:
            return team_results[0][1], team_results[0][2]
        team_results.sort(key=lambda x: x[0])
        # tie check
        if len(team_results) > 1 and team_results[0][0] == team_results[1][0]:
            return [], None
        return team_results[0][1], team_results[0][2]

    def check_tie(self, players):
        """Check if all surviving teams have identical averages."""
        avgs = []
        for team in self.teams:
            members = [p for p in players if p.name in team]
            if members:
                avg = sum(p.last_bidding for p in members) / len(members)
                avgs.append(avg)
        return len(avgs) > 1 and len(set(avgs)) == 1

    def run_single_round(self, round_id):
        # 1) Start round for all players
        for p in self.survival_players:
            if self.is_collaborative(p):
                p.start_round(round_id)
            else:
                # solo start_round accepts history
                p.start_round(round_id, self.full_round_history)

        # 2) Separate players by type
        collab = [p for p in self.survival_players if self.is_collaborative(p)]
        solo = [p for p in self.survival_players if not self.is_collaborative(p)]

        # 3) Collaborative pipeline for collaborative players only
        # Group collaborative players by their team index
        collab_teams = {i: [p for p in collab if p.name in team]
                        for i, team in enumerate(self.teams)}
        # Initial proposals
        initial = {}
        for idx, members in collab_teams.items():
            if not members:
                continue
            for p in members:
                p.act_initial(round_id, self.full_round_history)
            initial[idx] = {p.name: p.initial_bid for p in members}
        # Feedback stage
        feedback = {idx: {} for idx in collab_teams}
        for idx, members in collab_teams.items():
            for p in members:
                # summarize teammates' bids
                others = [f"{name} proposed {bid}" for name, bid in initial[idx].items() if name != p.name]
                summary = "\n".join(others)
                fb = p.act_feedback(round_id, summary)
                feedback[idx][p.name] = fb
        # Distribute feedback
        for idx, members in collab_teams.items():
            for p in members:
                summary = "".join(
                    f"{other}'s feedback on your proposal:\n{fb}\n\n"
                    for other, fb in feedback[idx].items() if other != p.name
                )
                if summary.strip():
                    p.receive_teammates_feedback(round_id, summary)
        # Final bids for collaborative players
        for p in collab:
            p.act_final(round_id)

        # 4) Solo players make their bids
        for p in solo:
            p.act()

        # 5) Compile bids and compute target
        bids = {p.name: p.last_bidding for p in self.survival_players}
        avg = sum(bids.values()) / len(bids)
        target = round_number(avg * 0.8, 2)

        # 6) Record history
        self.full_round_history.append({
            "round_id": round_id,
            "bids": bids,
            "target": target
        })

        # 7) Determine winner / tie
        if self.check_tie(self.survival_players):
            winner, winner_avg = [], None
        else:
            winner, winner_avg = self.check_winner(self.survival_players, target)
        self.round_winner[round_id] = winner

        # 8) Deduct HP
        self.round_deduction(self.survival_players, winner)

        # 9) Prepare messaging
        bidding_details = ", ".join(f"{n} chose {b}" for n, b in bids.items())
        diffs = ", ".join(
            f"{p.name}: |{p.last_bidding} - {target}| = {round_number(abs(p.last_bidding - target))}"
            for p in self.survival_players
        )
        infos = ", ".join(p.show_info() for p in self.survival_players)
        nums = " + ".join(str(p.last_bidding) for p in self.survival_players)

        if not winner and winner_avg is None:
            # Tie
            info_text = (
                f"Thank you all for participating in Round {round_id}. In this round, {bidding_details}.\n"
                f"All teams are tied for closeness to {target}. All players lose 1 HP. After deduction: {infos}."
            )
        else:
            winner_str = ", ".join(winner)
            info_text = (
                f"Thank you all for participating in Round {round_id}. In this round, {bidding_details}.\n"
                f"The average is ({nums}) / {len(self.survival_players)} = {avg}.\n"
                f"0.8 * average = {target}.\n"
                f"{diffs}\n"
                f"Team members {winner_str} win with average {winner_avg}, closest to {target}. "
                f"Others lose 1 HP. After deduction: {infos}."
            )

        # 10) Notify players and prune deaths
        next_survivors = []
        dead = []
        for p in self.survival_players:
            win_flag = p.name in winner
            p.notice_round_result(round_id, info_text, target, win_flag, bidding_details,
                                   {n: pl.biddings for n, pl in zip(bids.keys(), self.survival_players)})
            if p.hp <= 0:
                dead.append(p)
            else:
                next_survivors.append(p)
        self.survival_players = next_survivors

        # message eliminations
        for out in dead:
            for p in self.survival_players:
                if getattr(p, 'is_agent', False):
                    p.message.append({
                        "role": "system",
                        "content": f"{out.name}'s HP fell to zero and they have been eliminated."
                    })
        # end round
        for p in self.survival_players:
            p.end_round()

        print(f"Round {round_id}: {bidding_details}")

    def run_multi_round(self, max_round):
        for p in self.all_players:
            p.ROUND_WINNER = self.round_winner
        for i in range(1, max_round + 1):
            self.run_single_round(i)
