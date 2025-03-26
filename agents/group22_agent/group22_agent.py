import logging
from random import randint
from time import time
from typing import cast
from decimal import Decimal
import numpy as np
import builtins

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger
from geniusweb.bidspace.pareto.ParetoLinearAdditive import ParetoLinearAdditive

from .utils.opponent_model import OpponentModel


class TemplateAgent(DefaultParty):
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
        self.received_bids: list[(Bid, float)] = []
        self.sent_bids: list[(Bid, float)] = []
        self.opponent_utilities = []
        self.agent_utilities = []
        self.opponent_model: OpponentModel = None
        self.reservation_utility: Decimal = None
        self.sorted_bids = []
        self.logger.log(logging.INFO, "party is initialized")

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
        return "Template agent for the ANL 2022 competition"

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

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        bid = self.find_bid()
        # print(self.progress.get(time() * 1000))
        if self.last_received_bid is not None:
            self.opponent_utilities.append(Decimal(self.opponent_model.get_predicted_utility(self.last_received_bid)))
            self.received_bids.append((self.last_received_bid, self.progress.get(time() * 1000)))
        if self.accept_condition(self.last_received_bid, bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            action = Offer(self.me, bid)
            self.sent_bids.append((bid, self.progress.get(time() * 1000)))
            self.agent_utilities.append(Decimal(self.profile.getUtility(bid)))

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

    ###########################################################################################
    ###################################### Agent's logic ######################################
    ###########################################################################################

    def accept_condition(self, bid: Bid, next_bid: Bid) -> bool:
        reservation_utility = 0
        if self.profile.getReservationBid() is not None:
            reservation_utility = self.profile.getUtility(self.profile.getReservationBid())
        current_utility = 0
        if bid is not None:
            current_utility = self.profile.getUtility(bid)
        if current_utility < reservation_utility:
            return False
        if next_bid is not None:
            next_utility = self.profile.getUtility(next_bid)
        else:
            next_utility = 1
        # print(f"Utility of next bid: {next_utility}")
        # print(f"Utility of offered bid: {current_utility}")
        ACnext_res = next_utility <= current_utility
        if ACnext_res:
            return True
        else:
            progress = self.progress.get(time() * 1000)
            if len(self.received_bids) > 1:
                last_time_diff = self.received_bids[-1][1] - self.received_bids[-2][1]
                if 1 - progress < last_time_diff:
                    return True
            if progress > 0.98:
                agent_utility_nash, opponent_utility_nash = self.calculate_nash_point(self.sorted_bids)
                return agent_utility_nash-current_utility <= 0.1
                # # Compare to the max of the previous 2%
                # max_utility = 0
                # for b, t in self.received_bids:
                #     if t > progress - (1-progress):
                #         max_utility = max(max_utility, self.profile.getUtility(b))
                # return current_utility >= max_utility
            return False

    def find_bid(self) -> Bid:
        # compose a list of all possible bids
        if len(self.opponent_utilities) <= 1:
            domain = self.profile.getDomain()
            self.all_bids = AllBidsList(domain)
            self.sorted_bids = sorted(self.all_bids, key=lambda x: self.profile.getUtility(x))
            return self.sorted_bids[-1]
        best_bid = None
        smallest_diff = np.inf
        agent_utility_nash, opponent_utility_nash = self.calculate_nash_point(self.sorted_bids)
        # print(f"Agent utiltiy nash: {agent_utility_nash}")
        # print(f"Opponent utiltiy nash: {opponent_utility_nash}")
        if self.opponent_utilities[-2] == opponent_utility_nash or self.progress.get(time() * 1000) < 0.50:
            return self.sent_bids[-1][0]
        if self.progress.get(time() * 1000) > 0.98:
            max = 0
            bid = None
            for b, t in self.received_bids:
                if b is None:
                    continue
                cu = self.profile.getUtility(b)
                if cu > max:
                    bid = b
                    max = cu
            return bid
        # print(f"Previous opponent utility: {self.opponent_utilities[-1]}")
        # print(f"Second previous opponent utility: {self.opponent_utilities[-2]}")
        utility_diff = (self.opponent_utilities[-2] - self.opponent_utilities[-1]) / (self.opponent_utilities[-2] - opponent_utility_nash)
        utility_diff = Decimal(builtins.max(-0.2, min(utility_diff, 0.2)))
        # print(f"We think the opponent has made this progress towards the nash: {utility_diff}")
        if agent_utility_nash == self.agent_utilities[-1]:
            return self.sent_bids[-1][0]
        for _ in range(500):
            bid = self.sorted_bids[(randint(0, len(self.sorted_bids) - 1))]
            bid_utility = self.profile.getUtility(bid)
            diff = (self.agent_utilities[-1] - bid_utility) / (self.agent_utilities[-1] - agent_utility_nash)
            if abs(diff - utility_diff) < smallest_diff:
                best_bid = bid
                smallest_diff = abs(diff - utility_diff)
        return best_bid

        # best_bid_score = 0.0
        # best_bid = None
        #
        # # take 500 attempts to find a bid according to a heuristic score
        # for _ in range(500):
        #     bid = all_bids.get(randint(0, all_bids.size() - 1))
        #     bid_score = self.score_bid(bid)
        #     if bid_score > best_bid_score:
        #         best_bid_score, best_bid = bid_score, bid
        #
        # return best_bid

    def calculate_nash_point(self, all_bids):
        nash_point = 0
        agent_utility_nash, opponent_utility_nash = 0, 0
        for _ in range(500):
            bid = all_bids[(randint(0, len(all_bids) - 1))]
            agent_score = self.profile.getUtility(bid)
            opponent_score = Decimal(self.opponent_model.get_predicted_utility(bid))
            if agent_score * opponent_score > nash_point and agent_score > 0.7:
                nash_point = agent_score * opponent_score
                agent_utility_nash, opponent_utility_nash = agent_score, opponent_score
        return agent_utility_nash, opponent_utility_nash

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)

        our_utility = float(self.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score
