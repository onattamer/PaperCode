import os
import sys
import json
import argparse

# Ensure the project root (one level up) is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Also add Scenario1 and Scenario2 directories so their ollama_client modules can be imported
scenario1_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Scenario1'))
scenario2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Scenario2_version2'))
sys.path.insert(0, scenario1_dir)
sys.path.insert(0, scenario2_dir)

# Import the hybrid game runner (located in this folder)
from game_hybrid import G08AHybrid as G08A

# Import persona agents from both scenarios
from Scenario1.player.reasoning_player import PersonaAgentPlayer as SoloAgent
from Scenario2_version2.player.reasoning_player import PersonaAgentPlayer as CollabAgent

ENGINE = "qwen2.5:32b"


def load_personas(personas_file="personas.json"):
    with open(personas_file, "r") as f:
        data = json.load(f)
    return {p["name"]: p for p in data["personas"]}


def main(args):
    # Which team uses collaboration? 1 or 2
    collab_team_index = args.collab_team - 1

    for exp_no in range(args.start_exp, args.start_exp + args.exp_num):
        persona_dict = load_personas(args.personas_file)

        # Determine team size
        if args.mode == "2vs2":
            team_size = 2
        elif args.mode == "3vs3":
            team_size = 3
        else:
            raise ValueError("Mode must be '2vs2' or '3vs3'.")

        # Parse persona lists
        team1 = [s.strip() for s in args.team1_personas.split(',')]
        team2 = [s.strip() for s in args.team2_personas.split(',')]
        if len(team1) != team_size or len(team2) != team_size:
            raise ValueError(f"Each team must have {team_size} personas.")

        teams = [team1, team2]

        # Build player objects
        players = []
        for team_idx, persona_names in enumerate(teams):
            for name in persona_names:
                if name not in persona_dict:
                    raise ValueError(f"Persona {name} not found.")
                info = persona_dict[name]
                if team_idx == collab_team_index:
                    player = CollabAgent(info['name'], info, ENGINE)
                else:
                    player = SoloAgent(info['name'], info, ENGINE)
                players.append(player)

        # Inject teammate info into each agent's context
        for team in teams:
            details = [
                {"name": persona_dict[n]['name'],
                 "personality": persona_dict[n]['personality'],
                 "occupation": persona_dict[n]['occupation'],
                 "background": persona_dict[n]['background']}
                for n in team
            ]
            for p in players:
                if p.name in [d['name'] for d in details]:
                    teammates = [d for d in details if d['name'] != p.name]
                    info_lines = [
                        f"- {mate['name']}: {mate['personality']} {mate['occupation']}, {mate['background']}"
                        for mate in teammates
                    ]
                    p.message.append({
                        "role": "system",
                        "content": f"**Your Teammates:**\n" + "\n".join(info_lines)
                    })

        # Run the hybrid game
        game = G08A(players, teams)
        game.run_multi_round(args.max_round)

        # Prepare output
        prefix = f"hybrid_exp{exp_no}_team{args.collab_team}_mode{args.mode}"
        out_path = os.path.join(args.output_dir, f"{prefix}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Collect results
        export = {'winners': game.round_winner, 'biddings': {}, 'message': {}, 'logs': {}}
        for p in game.all_players:
            if getattr(p, 'is_agent', False):
                export['message'][p.name] = p.message
                export['biddings'][p.name] = getattr(p, 'biddings', [])
                if hasattr(p, 'logs') and p.logs:
                    export['logs'][p.name] = p.logs

        # Write JSON
        with open(out_path, 'w') as fout:
            json.dump(export, fout, indent=4)

        print(f"Hybrid experiment {exp_no} completed. Results: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='3vs3',
                        help="Game mode: '2vs2' or '3vs3'")
    parser.add_argument('--team1_personas', type=str,
                        default='Rachel Stein, Robbie Cruz, Marina Lopez',
                        help="Comma-separated names for Team 1")
    parser.add_argument('--team2_personas', type=str,
                        default='Adrian Park, Roxanne Gray, Dominic Liu',
                        help="Comma-separated names for Team 2")
    parser.add_argument('--collab_team', type=int, choices=[1, 2], default=1,
                        help='Which team uses collaborative agents (1 or 2)')
    parser.add_argument('--personas_file', type=str, default='personas.json',
                        help='Path to persona definitions')
    parser.add_argument('--output_dir', type=str, default='result_hybrid')
    parser.add_argument('--max_round', type=int, default=10)
    parser.add_argument('--start_exp', type=int, default=0)
    parser.add_argument('--exp_num', type=int, default=10)
    args = parser.parse_args()
    main(args)