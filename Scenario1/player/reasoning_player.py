#reasoning_player.py
import os
import time
import re
from .basic_player import Player
from ollama_client import call_ollama
import logging
import json
LOG_DIRECTORY = "/home/guest/Desktop/code/Scenario1/logging"

class AgentPlayer(Player):
    is_agent = True
    GAME_SETTING = (
        "TEAM SURVIVAL CHALLENGE!!! You are playing as part of a team in this game. "
        "Each player (on all teams) must choose an integer between 1 and 100 in each round. "
        "First, the overall average of numbers chosen by *all* players is calculated. "
        "The round's TARGET number is then determined as (0.8 * this overall average). "
        "Next, the average number chosen by the members of *each team* is calculated. "
        "The TEAM whose team average is closest to the TARGET number wins the round. "
        "Members of the winning team DO NOT lose any HP. "
        "All players on the OTHER teams lose 1 HP. "
        "If there's a tie (two or more teams are equally close to the target, or all team averages are identical), "
        "then ALL players lose 1 HP. "
        "Coordination with your teammates is crucial for your team's success!"
    )

    GAME_EXAMPLE = (
        "Team-Based Example:\n"
        "Assume Team A has Player Alex and Player Bob. Team B has Player Cindy and Player David.\n"
        "Alex chooses 50, Bob chooses 60. (Team A)\n"
        "Cindy chooses 20, David chooses 30. (Team B)\n\n"
        "1. Calculate Overall Average: (50 + 60 + 20 + 30) / 4 = 160 / 4 = 40.\n"
        "2. Calculate Target: 0.8 * Overall Average = 0.8 * 40 = 32.\n"
        "3. Calculate Team A's Average: (50 + 60) / 2 = 55.\n"
        "4. Calculate Team B's Average: (20 + 30) / 2 = 25.\n"
        "5. Compare Team Averages to Target:\n"
        "   - Team A: |Team Average 55 - Target 32| = 23\n"
        "   - Team B: |Team Average 25 - Target 32| = 7\n"
        "6. Determine Winner: Team B's average (25) is closer to the Target (32) than Team A's average (55).\n"
        "7. Result: Team B wins. Cindy and David lose 0 HP. Alex and Bob (from Team A) lose 1 HP each.\n\n"
        "General Rules Reminder:\n"
        "Every player starts with an initial HP of 10 points. "
        "Once a player's HP reaches 0, they are eliminated. "
        "The game continues until only players from one team remain, or a maximum round limit is reached. "
        "Strive to coordinate with your team to make choices that get your TEAM AVERAGE closest to the target (0.8 * overall average) to maximize your team's survival!"
    )

    INQUIRY = (
        "Ok , {name}! Now is the ROUND {round}, and your HP is at {hp}. "
        "Please choose an integer between 1 and 100 for this round."
    )
   
    def __init__(self, name, persona, engine):
        self.name = name
        self.engine = engine
        self.hp = 10
        
        self.biddings = []
        self.persona = persona
        self.message = [{"role": "system", "content": self.persona + self.GAME_SETTING.format(NAME=self.name)}]
        
        self.logs = None

    def log_prompt_history(self, round):
        log_dir = os.path.join(LOG_DIRECTORY, self.name)
        os.makedirs(log_dir, exist_ok=True)

        # Create file path: /LOG_DIRECTORY/AgentName/round_01_prompt.json
        file_path = os.path.join(log_dir, f"round_{str(round).zfill(2)}_prompt.json")
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.message, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Logging failed for {self.name} in round {round}: {e}")

        
    def act(self):
        print(f"Player {self.name} conduct bidding")
        status = 0
        while status != 1:
            try:
                response = call_ollama(
                    messages=self.message,
                    model=self.engine,
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95
                )
                response_content = response["choices"][0]["message"]["content"]
                self.message.append({"role": "assistant", "content": response_content})
                status = 1
            except Exception as e:
                print(e)
                time.sleep(15)
        self.biddings.append(self.parse_result(response_content))
            
    def parse_result(self, message):
        status = 0
        times = 0
        while status != 1:
            try:
                response = call_ollama(
                    messages=[
                        {"role": "system", "content": "Extract only the number chosen by the player. Output format: Just the number with no other text"},
                        {"role": "user", "content": message}
                    ],
                    model=self.engine,
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95
                )
                response_content = response["choices"][0]["message"]["content"].strip()
                numbers = re.findall(r'\d+', response_content)
                if numbers:
                    bidding_info = int(numbers[0])
                    if 1 <= bidding_info <= 100:
                        status = 1
                        return bidding_info
                raise AssertionError("Invalid number found: " + message)
            except AssertionError as e:
                print("Result Parsing Error:", e)
                times += 1
                if times >= 3:
                    raise RuntimeError("Failed to parse a valid bid after 3 attempts.")
            except Exception as e:
                print(e)
                time.sleep(15)
        return None
                        
    def start_round(self, round_id, full_round_history=None):
        self.message.append({
            "role": "system",
            "content": self.INQUIRY.format(name=self.name, round=round_id, hp=self.hp)
        })

        if full_round_history and len(full_round_history) > 0:
            last_round = full_round_history[-1]
            history_lines = [f"Round {last_round['round_id']} Recap:"]
            for name, bid in last_round['bids'].items():
                deviation = round(abs(bid - last_round['target']), 2)
                history_lines.append(
                    f"- {name}: Bid {bid} | Target: {last_round['target']} | Deviation: {deviation}"
                )
            recap_message = "\n".join(history_lines)
            self.message.append({
                "role": "system",
                "content": f"Here is the previous round's bidding history:\n{recap_message}"
            })

        # ✅ Log what the agent sees at the start of the round
        self.log_prompt_history(round_id)

        
    def notice_round_result(self, round, bidding_info, round_target, win, bidding_details, history_biddings):
        self.message_update_result(bidding_info)
        self.message_update_warning(win)
        
    def message_update_result(self, bidding_info):
        self.message.append({"role": "system", "content": bidding_info})
    
    def message_update_warning(self, win):
        def add_warning():
            if not win:
                if self.hp < 5:
                    return f"WARNING: You have lost 1 point of HP in this round! You now have only {self.hp} points of health left. You are in DANGER and one step closer to death. "
                if self.hp <= 3:
                    return f"WARNING: You have lost 1 point of HP in this round! You now have only {self.hp} points of health left. You are in extreme DANGER and one step closer to death.  "
                return f"WARNING: You have lost 1 point of HP in this round! You now have only {self.hp} points of health left. You are one step closer to death.  "
            return "You have successfully chosen the number closest to the target number, which is the average of all players' selected numbers multiplied by 0.8. As a result, you have won this round. All other players will now deduct 1 HP. "
        
        self.message.append({"role": "system", "content": add_warning()})
        
    def conduct_inquiry(self, inquiry):
        while True:
            try:
                response = call_ollama(
                    messages=self.message + [{"role": "system", "content": inquiry}],
                    model=self.engine,
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.9
                )
                response_content = response["choices"][0]["message"]["content"]
                return response_content
            except Exception as e:
                print(e)
                time.sleep(15)
                
class PersonaAgentPlayer(AgentPlayer):
    INQUIRY_PERSONA = ("Ok , {name}! Now is the ROUND{round}, and your HP is at {hp}. "
                       "Please choose an integer between 1 and 100 for this round. "
                       "Remember your unique personality and approach to this game and use your strengths to maximize your chances of survival!")
    
    def __init__(self, name, persona, engine):
        # Initialize with an empty persona first.
        super().__init__(name, "", engine)
        
        # Construct a persona prompt from the provided details.
        persona_description = (
            f"You are {persona['name']}, a {persona['occupation']}. "
            f"Your personality: {persona['personality']} "
            f"Background: {persona['background']} "
        )
        
        self.persona = persona_description
        
        # Reinitialize the system message with updated persona details.
        self.message = [
            {"role": "system", "content": self.persona + self.GAME_SETTING.format(NAME=self.name) + "\n" + self.GAME_EXAMPLE}
        ]
    
    def start_round(self, round_id, full_round_history=None):
        self.message.append({
            "role": "system",
            "content": self.INQUIRY_PERSONA.format(name=self.name, round=round_id, hp=self.hp)
        })

        if full_round_history and len(full_round_history) > 0:
            last_round = full_round_history[-1]
            history_lines = [f"Round {last_round['round_id']} Recap:"]
            for name, bid in last_round['bids'].items():
                deviation = round(abs(bid - last_round['target']), 2)
                history_lines.append(
                    f"- {name}: Bid {bid} | Target: {last_round['target']} | Deviation: {deviation}"
                )
            recap_message = "\n".join(history_lines)
            self.message.append({
                "role": "system",
                "content": f"Here is the previous round's bidding history:\n{recap_message}"
            })
        self.log_prompt_history(round_id)
