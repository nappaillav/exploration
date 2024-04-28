class prompt_manager:
    def __init__(self, capacity=10):
      self.capacity = capacity
      self.past_info = []
      self.ptr = 0
      self.ActionTable = { 0 : "No action", 1 : "Move right", 2 : "Move right and jump", 3 : "Move right and sprint", 4 : "Move right, jump, and sprint", 5 : "Jump", 6 : "Move left" }
      self.action_prompt()
      self.prompt_intro()

    def action_prompt(self):
      self.action_description = """ Action table = { 0 : "No action", 1 : "Move right", 2 : "Move right and jump", 3 : "Move right and sprint", 4 : "Move right, jump, and sprint", 5 : "Jump", 6 : "Move left" } """

    def prompt_intro(self):
      self.intro = "You are an AI assistant trained to play the Super Mario Bros game. Your goal Mario take actions that maximizes the score / reward by collecting coins, reaching flags, and completing stages while avoiding death."

    def add_info(self, current_info, previous_info, action, reward, done):
      coins = current_info["coins"] - previous_info["coins"]
      flag = current_info["flag_get"]
      life = current_info["life"] - previous_info["life"]
      time = current_info["time"] - previous_info["time"]
      pos = current_info["x_pos"] - previous_info["x_pos"]
      diff_score = current_info["score"] - previous_info["score"]
      prev_position = str(previous_info["x_pos"], previous_info["y_pos"])
      current_position = str(current_info["x_pos"], current_info["y_pos"])
      action_str = self.ActionTable[action]

      flag_str, life_str, done_str = "", "", ""

      if flag == True:
        flag_str = "Agent reached the Flag!! Great news. "
      if life != 0:
        life_str = "Mario Lost a life. "
      if done:
        done_str = 'Episode ended because the agent lost its life. '

      prompt = f"Agent positioned at {prev_position} took an action {action} ({action_str}) and moved to {current_position}, change in time is {time}, collected {coins} coins, score change is {diff_score} and recieved a reward of {reward}. "
      prompt += flag_str + life_str + done_str

      if self.ptr == self.capacity:
        self.past_info.pop(0)
        self.past_info.append(prompt)
      else:
        self.past_info.append(prompt)
        self.ptr += 1
      return prompt
    def get_prompt(self):
      return self.prompt_intro +'\n'+ self.action_prompt + '\n'.join(self.past_info)