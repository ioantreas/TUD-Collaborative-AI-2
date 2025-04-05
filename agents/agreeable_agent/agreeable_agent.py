from decimal import Decimal
import logging
import random
from time import time
from typing import cast

import numpy as np

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.bidspace.BidsWithUtility import BidsWithUtility
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditive import LinearAdditive
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


class AgreeableAgent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

        self.bids_with_util = None
        self.pMin = 0.0
        self.pMax = 1.0
        self.k = None
        self.expectation_time = 0
        self.MINIMUM_UTILITY = 0.5
        self.RESERVATION_VALUE = 0.2

        # Default values used in the paper
        self.ALPHA = 0.05
        self.CONCESSION_FACTOR = 0.1
        self.last_utility = 0
        self.c1 = 0.1
        self.c2 = 0.25

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            self.expectation_time = 0.75 #TODO change based on domain
            self.bids_with_util = BidsWithUtility.create(cast(LinearAdditive, self.profile))
            self.pMin = float(self.bids_with_util.getRange().getMin())
            self.pMax = float(self.bids_with_util.getRange().getMax())
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Agreeable Agent implementation"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()
            if self.k is None:
                self.k = float(self.profile.getUtility(bid))

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        my_bid = self.make_next_bid()
        # check if the last received offer is good enough
        if self.accept_condition(my_bid, self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            action = Offer(self.me, my_bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    def accept_condition(self, my_bid: Bid, opponent_bid: Bid) -> bool:
        if opponent_bid is None:
            return False
        
        t = self.progress.get(time() * 1000) #considering time too
        my_utility = self.profile.getUtility(my_bid)
        opponent_utility = self.profile.getUtility(opponent_bid)

        if t >= 0.99: # Last offer
            return opponent_utility >= self.RESERVATION_VALUE

        if t <= self.expectation_time: # Learning time
            return opponent_utility >= min(0.9, Decimal(1 - self.c1*(t/self.expectation_time)) * my_utility)
        
        return opponent_utility >= min(0.9, Decimal(1+self.c2 - self.c2*(t - self.expectation_time)/(1 - self.expectation_time)) * my_utility)

    def make_next_bid(self) -> Bid:
        progress = self.progress.get(time() * 1000)
        target_utility = self.get_target_utility(progress)

        if target_utility < self.MINIMUM_UTILITY:
            target_utility = self.MINIMUM_UTILITY

        explorable_neighbourhood = self.get_explorable_neighbourhood(progress, target_utility)

        validBids =[]
        for bid in AllBidsList(self.domain):
            if self.profile.getUtility(bid) >= explorable_neighbourhood[0] and self.profile.getUtility(bid) <= explorable_neighbourhood[1]:
                validBids.append(bid)

        if not validBids:
            return AllBidsList(self.domain).get(0) #return random bid if no options

        if self.opponent_model is None:
            return validBids[0]

        return self.get_best_bid_by_opponent_utilities(validBids)
    
    def get_explorable_neighbourhood(self, progress: float, target_utility: float) -> list[float, float]:
        if progress < self.expectation_time or self.opponent_model is None:
            return [target_utility, target_utility]

        threshold = self.ALPHA * (1 - (self.pMin + (self.pMax - self.pMin) * (1 - self.F(progress))))
        return [max(0.0, target_utility - threshold), min(1.0, target_utility + threshold)]

    def get_target_utility(self, progress: float):
        if progress <= self.expectation_time:
            return 1
        t = progress - self.expectation_time
        return self.pMin + (self.pMax - self.pMin) * (1 - self.F(t))
    
    def F(self, t: float) -> float:
        return self.k + (1 + self.k) * (t**(1/self.CONCESSION_FACTOR))
    
    def get_best_bid_by_opponent_utilities(self, bids):
        """
        Select a bid from the neighborhood that achieves median improvement in opponent utility
        """
        opponent_utils = np.array([self.opponent_model.get_predicted_utility(bid) for bid in bids])
        good_opponent_utils_indices = [(i,u) for i, u in enumerate(opponent_utils) if u >= self.last_utility]
        if len(good_opponent_utils_indices) > 0:
            index = good_opponent_utils_indices[self.median_index([u for i, u in good_opponent_utils_indices])][0]
            self.last_utility = opponent_utils[index]
            return bids[index]
        random_bid = random.choice(bids)
        self.last_utility = self.opponent_model.get_predicted_utility(random_bid)
        return random_bid

    def median_index(self, lst):
        """
        Given a list, selects the index of the median
        """
        sorted_lst = sorted(lst)
        median_value = sorted_lst[(len(lst) - 1) // 2] 
        return lst.index(median_value)

    # def roulette_wheel_selection(self, bids):
    #     total_utility = sum(self.opponent_model.get_predicted_utility(bid) for bid in bids)
    #     probabilities = []
    #     if total_utility != 0:
    #         probabilities = [self.opponent_model.get_predicted_utility(bid) / total_utility for bid in bids]
        

    #     # Randomly select a bid based on probability distribution
    #     cumulative_prob = 0
    #     rand_value = random.random()
    #     for i, prob in enumerate(probabilities):
    #         cumulative_prob += prob
    #         if cumulative_prob >= rand_value:
    #             return bids[i]
    #     return bids[-1]  # Fallback to last bid
    