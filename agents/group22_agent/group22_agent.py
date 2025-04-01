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

from .utils.opponent_model import OpponentModel


class LeNegotiator(DefaultParty):
    """
    Group 22 agent, aka LeNegotiator
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
        self.logger.log(logging.INFO, "party is initialized")
        self.reservation_utility: Decimal = None
        self.last_received_bid: Bid = None
        self.sent_bids: list[(Bid, float)] = None
        self.agent_utilities: list[Decimal] = None
        self.opponent_model: OpponentModel = None
        self.sorted_bids: list[Bid] = None
        self.all_bids: AllBidsList = None
        self.hardlining_time: float = 0.75
        self.agent_nash_point: Decimal = None
        self.opponent_nash_point: Decimal = None

    def notifyChange(self, data: Inform):
        """
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be sent to your
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
            self.agent_utilities = []
            self.opponent_model = OpponentModel(self.domain)

            # store useful lists with all bids
            self.all_bids = AllBidsList(self.domain)
            self.sorted_bids = sorted(self.all_bids, key=lambda x: self.profile.getUtility(x))

            # set reservation bid utility
            self.reservation_utility = self._getUtility(self.profile.getReservationBid())

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

        # Finished will be sent if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """
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
        """
        Returns a description of the agent.

        Returns:
            str: Agent description
        """
        return "Group 22 agent"

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
            self.opponent_model.update(bid, self.progress.get(time() * 1000))
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn.
        It checks if the bid received from the opponent is acceptable. If it is, the agent accepts.
        Otherwise, the agent finds a new bid to propose as a counteroffer.
        The chosen action is then sent to the opponent.
        """

        # recalculate nash point with new bids, used in acceptance condition and finding new bids
        self.agent_nash_point, self.opponent_nash_point = self.calculate_nash_point()

        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            action = Accept(self.me, self.last_received_bid)

        # if not, propose a counteroffer
        else:
            bid = self.find_bid()
            action = Offer(self.me, bid)

            # update lists that keep track of bids, their time and utility
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

    def _getUtility(self, bid: Bid) -> Decimal:
        """This method returns 0 utility if bid is None, else call getUtility() method from profile.

        Args:
            bid (Bid): bid for which to calculate utility

        Returns:
            Decimal: calculated utility
        """

        if bid is None:
            return Decimal(0)
        return self.profile.getUtility(bid)

    def calculate_nash_point(self) -> (Decimal, Decimal):
        """Calculates the Nash point utilies for the agent and its opponent.
        This is done by randomly sampling 500 bids, and selecting the one with the highest multiplied utility for the
        agent and its opponent.
        Additionally, it is ensured that the agent's utility is greater than the opponent's or a linearly decreasing
        boundary. This not only makes the Nash point a good goal for the agent to go towards, but also avoids utility
        coordinates to be switched every round (since opponent_model is constantly updated).

        Returns:
            (Decimal, Decimal): the utilities of the agent and its opponent, defined by the found Nash point
        """

        nash_point = 0
        agent_utility_nash, opponent_utility_nash = 0, 0

        # check a random selection of 500 bids
        for _ in range(500):
            bid = self.all_bids.get((randint(0, self.all_bids.size() - 1)))
            agent_score = self.profile.getUtility(bid)
            opponent_score = Decimal(self.opponent_model.get_predicted_utility(bid))

            # over time, the agent accepts nash points with a lower utility score, decreasing linearly from 0.7 to 0.5
            time_dec = Decimal(0.7 - self.progress.get(time() * 1000) / 5)

            # accept a point if the agent utility is greater than the opponent's or the linearly decreasing boundary
            if agent_score * opponent_score > nash_point and (agent_score > opponent_score or agent_score > time_dec):
                nash_point = agent_score * opponent_score
                agent_utility_nash, opponent_utility_nash = agent_score, opponent_score
        return agent_utility_nash, opponent_utility_nash

    def accept_condition(self, received_bid: Bid) -> bool:
        """Logic for acceptance condition of a received bid.
        Never accept if the utility is worse than our reservation utility.
        At any point, if the bid is better than our last offer times a time-based decay factor, accept.
        At the start of the game, when we have no sent bids to compare to, simply accept if we consider the offer to be
        very good - utility is above 0.8. From empirical testing receiving such an offer would be very fortunate.
        In the last 5% of the game, we are more willing to accept due to time pressure. As such, we accept if the offer
        is no worse than 0.15 away from the agent's Nash point utility.
        If we believe that the opponent will not have time for another round of offers, simply accept whatever bid they
        offered since it is still better than our reservation utility.

        Args:
            received_bid (Bid): bid received from the opponent

        Returns:
            bool: True if we accept the opponent's bid, False otherwise
        """

        received_utility = self._getUtility(received_bid)

        # base condition: if the received bid is less than the utility of the reservation bid, reject offer
        if received_utility < self.reservation_utility:
            return False

        # if we have not yet sent any bids, simply accept only if the utility of the received offer is equal to the best offer we can get
        if self.sent_bids is None:
            self.sent_bids = []
            if self.sorted_bids:
                return received_utility == self.sorted_bids[-1]
            return False

        # calculate a multiplication factor that linearly decreases with time, from 0 to 0.9
        progress = self.progress.get(time() * 1000)
        multiplication_factor = Decimal(1 - progress / 10)

        # check if the last offer times a small acceptance buffer was worse than the received utility, if so, accept
        if Decimal(self.agent_utilities[-1]) * multiplication_factor <= received_utility:
            return True

        # check if there is time remaining for another round of offers, if not, simply accept what we have.
        # We want to avoid failed negotiations as much as possible
        if len(self.opponent_model.offers) > 1:
            last_time_diff = self.opponent_model.offers[-1][1] - self.opponent_model.offers[-2][1]
            if 1 - progress < last_time_diff:
                return True

        # in the final stage of negotiation, accept offers that are no more than 0.15 worse than our nash point utility
        if progress > 0.95:
            if self.agent_nash_point - received_utility <= 0.15:
                return True

        # if all other conditions fail, do not accept
        return False

    def find_bid(self) -> Bid:
        """Finds the next bid to be made by the agent.
        For the time specified by the hardlining_time parameter, the agent randomly selects a bid from a range of sorted
        bids. This range linearly increases with time, and will include a maximum of 76 elements.
        Thereafter, this agent uses tit-for-tat negotiation by mirroring the opponent's utility concessions relative
        to the Nash product, doing so by trying to find the bid that is closest to the utility_diff parameter.
        If we believe there is not enough time for us to make another offer, we return the bid with the highest
        utility that was made by their opponent (as long as it is better than the utility of our reservation bid),
        since it will have a high change of being accepted since it was originally proposed by the opponent. If their
        best bid is still below the reservation utility, we simply send our reservation bid as a last resort.

        Returns:
            Bid: the next bid to be made by the agent
        """

        progress = self.progress.get(time() * 1000)

        # if we have not received a bit yet, simply send the bid with the best utility for us
        if len(self.opponent_model.offer_utilities) <= 1:
            return self.sorted_bids[-1]

        # if we are being a hardliner, send a random bid from a range of sorted bids that slowly increases over time
        # range is at least the best element, and linearly increase over time to include at most the best 76 elements
        if self.progress.get(time() * 1000) < self.hardlining_time:
            max_index = len(self.sorted_bids) - 1
            min_index = int(max_index - progress * 100)
            return self.sorted_bids[(randint(min_index, max_index))]

        # if this is going to be our last bid, send the offer the opponent made that gave us the best utility, as long
        # as it is better than the reservation bid. Otherwise, send the reservation bid as a last resort
        last_time_diff = self.sent_bids[-1][1] - self.sent_bids[-2][1]
        if 1 - progress < last_time_diff:
            best_opponent_bid = sorted(self.opponent_model.offers, key=lambda x: self.profile.getUtility(x[0]))[-1][0]
            if self.profile.getUtility(best_opponent_bid) < self.reservation_utility:
                return self.profile.getReservationBid()
            return best_opponent_bid

        # calculate the concession made by the opponent in terms of their movement relative to the nash product
        epsilon = Decimal(1e-9)  # Small constant to prevent division by zero
        denominator = (self.opponent_model.offer_utilities[-2] - self.opponent_nash_point) + epsilon
        utility_diff = (self.opponent_model.offer_utilities[-2] - self.opponent_model.offer_utilities[-1]) / denominator

        # only make concessions with relative utility between -0.2 and 0.2 to the nash product
        utility_diff = Decimal(builtins.max(-0.2, min(utility_diff, 0.2)))

        # select the bid that is closest to the mirrored concession to make, from a random pool of 500 bids
        best_bid = None
        smallest_diff = np.inf
        for _ in range(500):
            bid = self.sorted_bids[(randint(0, len(self.sorted_bids) - 1))]
            bid_utility = self.profile.getUtility(bid)
            epsilon = Decimal(1e-9)  # Small constant to prevent division by zero
            denominator = (self.agent_utilities[-1] - self.agent_nash_point) + epsilon
            diff = (self.agent_utilities[-1] - bid_utility) / denominator
            if abs(diff - utility_diff) < smallest_diff:
                best_bid = bid
                smallest_diff = abs(diff - utility_diff)
        return best_bid
