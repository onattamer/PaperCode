import os
import json
from player import *
from game import G08A
import argparse

ENGINE = "qwen2.5:32b"

def load_personas(personas_file="personas.json"):
    with open(personas_file, "r") as f:
        data = json.load(f)
    # data should have a key "personas" which is a list of persona dicts
    return {p["name"]: p for p in data["personas"]}

def main(args):

    # Loop over the experiment range
    for exp_no in range(args.start_exp, args.start_exp + args.exp_num):

        # Load all personas from a JSON file
        persona_dict = load_personas(args.personas_file)

        # Determine team size based on mode
        if args.mode == "2vs2":
            team_size = 2
        elif args.mode == "3vs3":
            team_size = 3
        else:
            raise ValueError("Mode must be either '2vs2' or '3vs3'.")

        # Parse team personas (names)
        team1_personas = [s.strip() for s in args.team1_personas.split(',')]
        team2_personas = [s.strip() for s in args.team2_personas.split(',')]

        # Validate number of personas based on team size
        if len(team1_personas) != team_size:
            raise ValueError(f"Team 1 must have exactly {team_size} personas.")
        if len(team2_personas) != team_size:
            raise ValueError(f"Team 2 must have exactly {team_size} personas.")

        # Build player objects from persona names
        players = []
        for persona_name in team1_personas:
            if persona_name not in persona_dict:
                raise ValueError(f"Persona {persona_name} not found in persona list.")
            p_info = persona_dict[persona_name]
            p = PersonaAgentPlayer(p_info["name"], p_info, ENGINE)
            players.append(p)

        for persona_name in team2_personas:
            if persona_name not in persona_dict:
                raise ValueError(f"Persona {persona_name} not found in persona list.")
            p_info = persona_dict[persona_name]
            p = PersonaAgentPlayer(p_info["name"], p_info, ENGINE)
            players.append(p)

        # Define teams
        teams = [
            [persona_dict[n]["name"] for n in team1_personas],
            [persona_dict[n]["name"] for n in team2_personas]
        ]

        # Provide teammate information to each player
        for team in teams:
            team_details = [
                {
                    "name": persona_dict[persona]["name"],
                    "personality": persona_dict[persona]["personality"],
                    "occupation": persona_dict[persona]["occupation"],
                    "background": persona_dict[persona]["background"]
                }
                for persona in team
            ]
            for player in players:
                if player.name in [member["name"] for member in team_details]:
                    # Exclude the player's own details
                    teammates = [
                        member for member in team_details if member["name"] != player.name
                    ]
                    teammates_info = "\n".join([
                        f"- {mate['name']}: {mate['personality']} {mate['occupation']}, {mate['background']}"
                        for mate in teammates
                    ])
                    player.message.append({
                        "role": "system",
                        "content": f"**Your Teammates:**\n{teammates_info}"
                    })

        # Run the game
        Game = G08A(players, teams)
        Game.run_multi_round(args.max_round)

        # Construct a filename that includes the experiment number
        team1_str_joined = "_".join(team1_personas)
        team2_str_joined = "_".join(team2_personas)
        prefix = f"{team1_str_joined}_VS_{team2_str_joined}_exp{exp_no}"

        output_file = f"{args.output_dir}/{prefix}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Export game records
        with open(output_file, "w") as fout:
            messages = {}
            biddings = {}
            logs = {}
            for agent in Game.all_players:
                if agent.is_agent:
                    messages[agent.name] = agent.message
                biddings[agent.name] = agent.biddings
                if agent.logs:
                    logs[agent.name] = agent.logs

            debug_info = {
                "winners": Game.round_winner,
                "biddings": biddings,
                "message": messages,
                "logs": logs
            }

            json.dump(debug_info, fout, indent=4)

        print(f"Experiment {exp_no} completed. Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Mode remains the same
    parser.add_argument('--mode', type=str, default="3vs3",
                        help="Game mode: '2vs2' or '3vs3'")

    # Replace strategy arguments with persona arguments
    parser.add_argument('--team1_personas', type=str,
                        default="Rachel Stein, Robbie Cruz, Marina Lopez",
                        help="Comma-separated list of persona names for Team 1 players")
    parser.add_argument('--team2_personas', type=str,
                        default="Adrian Park, Roxanne Gray, Dominic Liu",
                        help="Comma-separated list of persona names for Team 2 players")

    # Path to the persona JSON file
    parser.add_argument('--personas_file', type=str, default="personas.json",
                        help="Path to the JSON file containing persona definitions")

    parser.add_argument("--output_dir", type=str, default="result")
    parser.add_argument("--init_mean", type=int, default=40,
                        help="(Unused now) init mean value for computer player")
    parser.add_argument("--norm_std", type=int, default=5,
                        help="(Unused now) standard deviation for computer players")

    parser.add_argument('--max_round', type=int, default=10)

    # Experiment loop parameters
    parser.add_argument('--start_exp', type=int, default=0,
                        help="Start experiment index")
    parser.add_argument('--exp_num', type=int, default=10,
                        help="Number of experiments to run")

    parser.add_argument('--player_engine', type=str, default=None,
                        help="player's OpenAI api engine")
    parser.add_argument('--player_k', type=int, default=None,
                        help="player's k-level (default 2)")

    args = parser.parse_args()
    main(args)
