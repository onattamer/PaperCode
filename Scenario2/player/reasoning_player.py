import os
import time
import re
# Assuming .basic_player exists relative to this file and defines Player class
from .basic_player import Player
# Assuming ollama_client.py exists and defines call_ollama function
from ollama_client import call_ollama
 
# Keep round_number alias if used elsewhere (often used in simulation logic)
round_number = round
 
class AgentPlayer(Player):
    """
    Represents an AI agent player in the G0.8A game using an LLM.
    Manages conversation history and context for LLM interaction.
    """
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
        """
        Initializes the AgentPlayer.
 
        Args:
            name (str): The player's name.
            persona (str): A string describing the agent's persona (can be empty for base AgentPlayer).
            engine (str): The identifier for the LLM engine to use.
        """
        super().__init__(name)
        self.engine = engine
        self.hp = 10 # Starting HP
        self.persona = persona # Store the persona description
        self.message = [
            {
                "role": "system",
                # Combine persona (if provided) and game settings for initial context
                "content": (self.persona + " " if self.persona else "") + self.GAME_SETTING.format(NAME=self.name)
            }
        ]
        self.logs = None # Placeholder for potential future logging
        # Marker for the start index of current round's internal interaction messages in self.message
        self._current_round_interaction_start_index = -1
        # Flag to track if history was added this round (for act_feedback context)
        self._history_added_this_round = False
 
    def parse_result(self, message):
        """
        Parses the LLM response to extract the chosen integer bid.
 
        Args:
            message (str): The content of the LLM's response message.
 
        Returns:
            int or None: The extracted integer bid (1-100), or None if parsing fails after retries.
 
        Raises:
            RuntimeError: If a valid bid cannot be parsed after multiple attempts.
        """
        status = 0
        times = 0
        max_retries = 3 # Max retries for parsing errors
        max_api_retries = 5 # Max retries for API/network errors
 
        while status != 1:
            try:
                # Use a separate, minimal context for number extraction
                extraction_context = [
                    {
                        "role": "system",
                        "content": "Extract only the first integer between 1 and 100 mentioned by the user. Output format: Just the integer number with no other text, explanation, or punctuation."
                    },
                    {"role": "user", "content": message}
                ]
                response = call_ollama(
                    messages=extraction_context,
                    model=self.engine,
                    temperature=0.1, # Low temperature for deterministic extraction
                    max_tokens=10, # Small limit for just the number
                    top_p=0.95, # Standard values for other params
                    frequency_penalty=0,
                    presence_penalty=0
                )
                response_content = response["choices"][0]["message"]["content"].strip()
                # More robust regex to find the first sequence of digits
                numbers = re.findall(r'\b\d+\b', response_content)
                if numbers:
                    # Try converting the first found number
                    bid_val = int(numbers[0])
                    if 1 <= bid_val <= 100:
                        status = 1
                        return bid_val
                    else:
                         # Found a number, but it's out of range
                         raise AssertionError(f"Number {bid_val} out of range (1-100).")
                # No valid number found in the response
                raise AssertionError("No valid integer (1-100) found.")
 
            except AssertionError as e:
                print(f"Result Parsing Error for {self.name} (Attempt {times + 1}/{max_retries}): {e} | Raw Response: '{response_content}' | Original Message: '{message[:100]}...'")
                times += 1
                if times >= max_retries:
                    print(f"FATAL: Failed to parse valid bid for {self.name} after {max_retries} attempts.")
                    # Optionally, return a default value or re-raise
                    # return 50 # Example default
                    raise RuntimeError(f"Failed to parse a valid bid for {self.name} after {max_retries} attempts.")
                time.sleep(2) # Short delay before retrying parse
 
            except Exception as e:
                # Handle potential API errors (network, rate limits, etc.)
                print(f"API/Network Error during bid parsing for {self.name} (Attempt {times + 1}): {e}")
                times += 1
                if times >= max_api_retries:
                    print(f"FATAL: API/Network error for {self.name} after {max_api_retries} attempts.")
                    raise RuntimeError(f"API/Network error for {self.name} after multiple attempts.")
                # Exponential backoff could be implemented here
                time.sleep(10 + (times * 5)) # Basic increasing delay before retrying API call
 
        return None # Should not be reached if RuntimeError is raised
 
    def start_round(self, round_id):
        """Adds the round start inquiry and marks the start for context pruning."""
        start_msg_content = self.INQUIRY.format(name=self.name, round=round_id, hp=self.hp)
        start_msg = {"role": "system", "content": start_msg_content}
        self.message.append(start_msg)
        # Mark the index *after* this message as the start of this round's interactions
        self._current_round_interaction_start_index = len(self.message)
        self._history_added_this_round = False # Reset history flag
 
    def notice_round_result(self, round, bidding_info, round_target, win, bidding_details, history_biddings):
        """Adds round results and resets the interaction marker."""
        self.message_update_result(bidding_info)
        self.message_update_warning(win)
        # Reset marker AFTER all round-end messages are added
        self._current_round_interaction_start_index = -1
        self._history_added_this_round = False
 
    def message_update_result(self, bidding_info):
        """Appends the round result information to the message history."""
        self.message.append({"role": "system", "content": bidding_info})
 
    def message_update_warning(self, win):
        """Appends a win/loss/warning message based on the round outcome."""
        def add_warning():
            # Constructs the appropriate warning message based on HP and win status
            if not win:
                if self.hp <= 3: # Check extreme danger first
                     return f"WARNING: You have lost 1 point of HP in this round! You now have only {self.hp} points of health left. You are in extreme DANGER and one step closer to death. Your next choices are critical!"
                elif self.hp < 5:
                     return f"WARNING: You have lost 1 point of HP in this round! You now have only {self.hp} points of health left. You are in DANGER and one step closer to death. Coordinate carefully!"
                else:
                    return f"WARNING: You have lost 1 point of HP in this round! You now have only {self.hp} points of health left. You are one step closer to death."
            else:
                 return "You have successfully chosen the number closest to the target number as a team, which is the average multiplied by 0.8. You won this round. All other players that are not in your team lose 1 HP."
 
        self.message.append({"role": "system", "content": add_warning()})
 
    # --- ACTION METHODS WITH CONTEXT PRUNING AND IMPROVED PROMPTING ---
 
    def act_initial(self, round_id, full_round_history):
        """Generates the agent's initial bid proposal for the round."""
        # Context: History up to the start of this round's interactions
        if self._current_round_interaction_start_index == -1:
             print(f"Warning: _current_round_interaction_start_index not set for {self.name} in round {round_id}. Using full history for safety.")
             local_context_base = self.message[:]
        else:
             # Slice up to the message *before* the interaction marker
             local_context_base = self.message[:self._current_round_interaction_start_index]
 
        local_context = list(local_context_base) # Temporary context for the API call
 
        # 1. Add Previous round history recap (if applicable)
        if full_round_history:
            last_round = full_round_history[-1]
            history_lines = [f"Round {last_round['round_id']} Recap:"]
            for name, bid in last_round['bids'].items():
                # Maybe exclude self from recap? Optional.
                # if name == self.name: continue
                deviation = round(abs(bid - last_round['target']), 2)
                history_lines.append(f"- {name}: Bid {bid} | Target: {last_round['target']} | Deviation: {deviation}")
            history_summary = "\n".join(history_lines)
            history_message = {
                "role": "system",
                "content": f"Here is the previous round's bidding history:\n{history_summary}"
            }
            local_context.append(history_message) # Add to local context for this call
            # Add history persistently *at the marker index* if not already there
            if self._current_round_interaction_start_index != -1 and \
               (len(self.message) <= self._current_round_interaction_start_index or \
                self.message[self._current_round_interaction_start_index] != history_message):
                 self.message.insert(self._current_round_interaction_start_index, history_message)
                 self._history_added_this_round = True # Set flag
 
        # 2. Prepare the specific action prompt for this step
        action_prompt = {
            "role": "system",
            "content": (
                f"**Action: Initial Proposal (Round {round_id})**\n"
                f"Carefully review the game rules and any past round history provided.\n"
                f"Your task *now* is ONLY to propose your FIRST bid for this round.\n"
                "Propose a single integer between 1 and 100. Provide a very short explanation for this *initial* choice.\n"
                "This proposal will be shared with your teammates for feedback later. Do NOT discuss feedback or final decisions yet."
            )
        }
        local_context.append(action_prompt) # Add to local context for this call
 
        # --- Logging for Debugging (Optional) ---
        # print(f"\nDEBUG: {self.name} - act_initial - Context Sent ({len(local_context)} msgs):")
        # for i, msg in enumerate(local_context):
        #     print(f"  [{i}] {msg['role']}: {msg['content'][:150].replace(os.linesep, ' ')}...")
        # ---
 
        # --- Call LLM ---
        response_content = call_ollama(messages=local_context, model=self.engine)["choices"][0]["message"]["content"].strip()
 
        # --- Update Persistent History ---
        # Append only the assistant's response to the main history
        self.message.append({"role": "assistant", "content": response_content})
        # ---
 
        # Parse the bid *after* adding the response to history
        try:
            self.initial_bid = self.parse_result(response_content)
        except RuntimeError as e:
            print(f"Error setting initial_bid for {self.name}: {e}. Setting default 50.")
            self.initial_bid = 50 # Default value if parsing fails completely
 
 
    def act_feedback(self, round_id, proposals_summary):
        """Generates feedback on teammates' initial proposals."""
        # Context: History up to the start of this round's interactions + history recap if added
        if self._current_round_interaction_start_index == -1:
             print(f"Warning: _current_round_interaction_start_index not set for {self.name} in round {round_id}. Using full history for safety.")
             local_context_base = self.message[:]
        else:
             # Slice up to the message *before* the interaction marker
             context_end_index = self._current_round_interaction_start_index
             # If history was added, include it in the base context sent to LLM
             if self._history_added_this_round:
                 # Check bounds before accessing index
                 if len(self.message) > context_end_index:
                     context_end_index += 1
                 else:
                      print(f"Warning: History flag set for {self.name} but message index {context_end_index} out of bounds.")
             local_context_base = self.message[:context_end_index]
 
        local_context = list(local_context_base) # Temporary context for the API call
 
        # 1. Prepare the specific action prompt for THIS action
        action_prompt = {
            "role": "system",
            "content": (
                f"**Action: Provide Feedback (Round {round_id})**\n"
                # Reference that initial proposal was made, but don't include its content
                f"You previously made an initial proposal for this round (details not shown here). Now, review *only* your teammates' initial proposals provided below:\n"
                f"---\n{proposals_summary}\n---\n\n" # Added separators
                "Your task *now* is ONLY to provide brief feedback or commentary *on your teammates' proposals*.\n"
                "If you believe a teammate should adjust their choice (higher or lower), explicitly advise *them*.\n"
                "Remind them that coordination is key and not adapting might lead to HP loss if the average shifts unexpectedly.\n"
                "**CRITICAL: Do NOT state your own final bid or revise your own initial proposal in this response. Focus entirely on commenting on your teammates' ideas.**"
            )
        }
        local_context.append(action_prompt) # Add to local context for this call
 
        # --- Logging for Debugging (Optional) ---
        # print(f"\nDEBUG: {self.name} - act_feedback - Context Sent ({len(local_context)} msgs):")
        # for i, msg in enumerate(local_context):
        #     print(f"  [{i}] {msg['role']}: {msg['content'][:150].replace(os.linesep, ' ')}...")
        # ---
 
        # --- Call LLM ---
        response_content = call_ollama(messages=local_context, model=self.engine)["choices"][0]["message"]["content"].strip()
 
        # --- Update Persistent History ---
        # Append only the assistant's response
        self.message.append({"role": "assistant", "content": response_content})
        # ---
 
        return response_content # Return the generated feedback text
 
    def receive_teammates_feedback(self, round_id, feedback_summary):
        """Adds the feedback received from teammates to the message history."""
        # This just adds info to history, no LLM call.
        feedback_message = {
            "role": "system",
            "content": (
                f"**Action: Receive Feedback (Round {round_id})**\n"
                f"Read the following feedback your teammates provided about *your* initial proposal for Round {round_id}:\n\n"
                f"---\n{feedback_summary}\n---\n\n" # Added separators
                "**CRITICAL REMINDER: Ignoring constructive feedback has often led to HP loss in past rounds! Carefully consider these points. You will make your final decision in the next step.**"
            )
        }
        self.message.append(feedback_message) # Add feedback to persistent history
 
 
    def act_final(self, round_id):
        """Generates the agent's final bid for the round after discussion."""
        # Context: Full history is needed here to reflect on the entire round's discussion.
        local_context_base = self.message[:] # Use the full persistent history
 
        local_context = list(local_context_base) # Temporary context for the API call
 
        # Prepare the specific action prompt for this final action
        action_prompt = {
            "role": "system",
            "content": (
                f"**Action: Final Decision (Round {round_id})**\n"
                f"This is your *final commitment* for Round {round_id}.\n"
                "Review the entire discussion for this round (your initial proposal, the feedback you gave, the feedback you received).\n"
                "**Explicitly state how the feedback you received and the overall discussion influenced your final choice.** Did you adjust your initial idea based on feedback? Why or why not?\n"
                "Now, choose your single, final integer bid between 1 and 100.\n"
                "Output format: Start with your final bid number (e.g., 'My final bid is 42.' or just '42'). Follow with your brief explanation incorporating reflection on the feedback and discussion."
            )
        }
        local_context.append(action_prompt) # Add prompt to context for the call
 
        # --- Logging for Debugging (Optional) ---
        # print(f"\nDEBUG: {self.name} - act_final - Context Sent ({len(local_context)} msgs):")
        # for i, msg in enumerate(local_context):
        #     print(f"  [{i}] {msg['role']}: {msg['content'][:150].replace(os.linesep, ' ')}...")
        # ---
 
        # --- Call LLM ---
        response_content = call_ollama(messages=local_context, model=self.engine)["choices"][0]["message"]["content"].strip()
 
        # --- Update Persistent History ---
        # Append only the assistant's response
        self.message.append({"role": "assistant", "content": response_content})
        # ---
 
        # Parse the bid *after* adding the response to history
        try:
            final_bid = self.parse_result(response_content)
            self.last_bidding = final_bid
        except RuntimeError as e:
             print(f"Error setting final_bid for {self.name}: {e}. Setting default 50.")
             self.last_bidding = 50 # Default value if parsing fails completely
 
 
# --- PersonaAgentPlayer Class ---
class PersonaAgentPlayer(AgentPlayer):
    """
    An AgentPlayer with a specific persona, background, and adjusted prompts.
    Inherits context pruning and action logic from AgentPlayer.
    """
    # Needs its own INQUIRY format if different from the base class
    INQUIRY_PERSONA = (
        # Ensure format matches AgentPlayer.INQUIRY if marker logic depends on it,
        # otherwise, adjust marker finding logic if formats differ significantly.
        # Assuming format "ROUND{round}" vs "ROUND {round}" might require adjustment in base class logic if used.
        "Ok , {name}! Now is the ROUND{round}, and your HP is at {hp}. "
        "Please choose an integer between 1 and 100 for this round, "
        "while reflecting your unique personality. Use your strengths to maximize survival!"
    )
 
    def __init__(self, name, persona, engine):
        """
        Initializes the PersonaAgentPlayer.
 
        Args:
            name (str): The player's name.
            persona (dict): A dictionary containing 'name', 'occupation', 'personality', 'background'.
            engine (str): The identifier for the LLM engine to use.
        """
        # Call parent __init__ - it sets up basic attributes including marker/flag
        super().__init__(name, "", engine) # Pass empty persona string to parent
 
        # Overwrite attributes specific to PersonaAgentPlayer
        self.engine = engine
        self.hp = 10
        persona_description = (
            f"You are {persona['name']}, a {persona['occupation']}. "
            f"Your personality: {persona['personality']} "
            f"Background: {persona['background']}"
        )
        # Store full persona text
        self.persona = persona_description
 
        # Rebuild message list for persona agent, including GAME_EXAMPLE
        self.message = [
            {
                "role": "system",
                "content": (
                    self.persona # Add detailed persona description first
                    + "\n\n" # Add separation
                    + self.GAME_SETTING.format(NAME=self.name)
                    + "\n\n---\n\n" # Add separation
                    + self.GAME_EXAMPLE # Add example
                )
            }
        ]
        # Reset marker and flag after rebuilding message list
        self._current_round_interaction_start_index = -1
        self._history_added_this_round = False
 
    # Override start_round to use INQUIRY_PERSONA and set marker/flag
    def start_round(self, round_id):
        """Adds the round start inquiry using INQUIRY_PERSONA and marks the start."""
        # Use the specific format for Persona agents
        start_msg_content = self.INQUIRY_PERSONA.format(name=self.name, round=round_id, hp=self.hp)
        start_msg = {"role": "system", "content": start_msg_content}
        self.message.append(start_msg)
        # Mark the index *after* this message as the start of this round's interactions
        self._current_round_interaction_start_index = len(self.message)
        self._history_added_this_round = False # Reset history flag