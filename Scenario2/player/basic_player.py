from random import randint
import numpy as np

class Player():
    def __init__(self, name):
        self.name = name
        self.hp = 10
        # We store bids in a list attribute
        self._biddings = []
        self.cur_round = -1
        
        self.logs = None
        
    def start_round(self, round_: int):
        self.cur_round = round_
    
    def act(self):
        raise NotImplementedError
    
    def notice_round_result(self, round_, bidding_info, round_target, win, bidding_details, history_biddings):
        raise NotImplementedError
    
    def end_round(self):
        pass
    
    def deduction(self, deducted_hp):
        self.hp -= deducted_hp

    @property
    def biddings(self):
        """
        If you need direct access to the entire list of bids,
        you can provide a separate property or method.
        """
        return self._biddings

    @property
    def last_bidding(self):
        """
        A property that returns the last element of biddings.
        If biddings is empty, return None.
        """
        if self._biddings:
            return self._biddings[-1]
        return None

    @last_bidding.setter
    def last_bidding(self, value):
        """
        Whenever we set 'last_bidding = X', append X to the biddings list.
        This ensures the final bid is always tracked in biddings.
        """
        self._biddings.append(value)

    def show_info(self, print_=False):
        if print_:
            print(f"NAME:{self.name}\tHEALTH POINT:{self.hp}\n")
        return f"NAME:{self.name}\tHEALTH POINT:{self.hp}"


class ProgramPlayer(Player):
    is_agent = False
    def __init__(self, name, strategy, mean, std):
        super().__init__(name)
        self.strategy = strategy
        self.mean = mean
        self.std = std
        
        self.logs = None
        
        if self.strategy == "monorand":
            self.std = randint(0, std)
            self.strategy = "mono"
            
    def start_round(self, round_):
        pass
    
    def end_round(self):
        if self.strategy == "mono":
            self.mean -= self.std
            
    def notice_round_result(self, round_, bidding_info, round_target, win, bidding_details, history_biddings):
        if self.strategy == "last":
            self.mean = round_target
            
    def set_normal(self, mean, std):
        self.normal = True
        self.mean = mean
        self.std = std
        
    def act(self):
        """
        A simple example strategy that picks a random or deterministic bid,
        then stores it using the property 'last_bidding'.
        """
        if self.strategy == "mono":
            bidding = self.mean
        else:
            bidding = np.random.normal(self.mean, self.std)

        # Ensure it's within [1..100]
        bidding = min(max(int(bidding), 1), 100)
        
        # Instead of self.biddings.append(...), we use the property:
        self.last_bidding = bidding
