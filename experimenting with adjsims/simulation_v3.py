""""
CSC 446 @ UVIC
Author: Sadie Johansen, V00715310
Date: 2023-12-03

*Github co-pilot used for this project*
"""

import random
import heapq
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from scipy import stats
import math
import time
import logging

import os

class FlowBranchOperator:
    def __init__(self, probabilities, children=None, origin=None):
        """
        Initialize a FlowBranchOperator object.

        Args:
            probabilities (list): A list of probabilities for each child.
            children (list, optional): A list of children. Defaults to None.
        """
        self.probabilities = probabilities
        # reduce children to only those with non-zero probability
        self.children = [] if children is None else [children[i] for i in range(len(children)) if self.probabilities[i] > 0]
        # reduce probabilities to only those with non-zero probability
        self.probabilities = [self.probabilities[i] for i in range(len(self.probabilities)) if self.probabilities[i] > 0]
        self.shortest_queue = False
        if np.sum(self.probabilities) > 1 and origin is not None:
            logging.info(f"{origin} branch method set as shortest queue")
            self.shortest_queue = True

    def randomly_select_child(self):
        try:
            return np.random.choice(self.children, p=self.probabilities)
        except:
            print(self.probabilities)
            print(self.children)
            raise ValueError("Probabilities do not sum to 1")  
        
    def get_children_ids(self):
        return self.children
    
    def uses_shortest_queue(self):
        return self.shortest_queue

    def is_sink(self):
        return sum(self.children) == 0 if self.children is not None else False

class Event:
    def __init__(self, event_type, time, server_id=None, source_id=None, event_id=None):
        """
        Initialize an Event object.

        Args:
            event_type (str): The type of the event.
            time (float): The time of the event.
            server_id (int, optional): The ID of the server associated with the event. Defaults to None.
            source_id (int, optional): The ID of the source associated with the event. Defaults to None.
            event_id (int, optional): The ID of the event. Defaults to None.
        """
        self.event_type = event_type
        self.time = time
        self.server_id = server_id
        self.source_id = source_id
        self.event_id = event_id
        self.delayed_event = False
        self.delayed_time = 0
        self.arrival_time = 0

    def __lt__(self, other):
        return self.time < other.time

    def get_type(self):
        return self.event_type

    def get_time(self):
        return self.time

    def get_server_id(self):
        return self.server_id
    
    def get_source_id(self):
        return self.source_id

    def get_event_id(self):
        return self.event_id
    

class EventList:
    def __init__(self):
        self.events = []
        self.servers_next_departure = {}

    def get_time_of_next_departure(self, server_id):
        if server_id not in self.servers_next_departure:
            return math.inf
        return self.servers_next_departure[server_id]

    def enqueue(self, event):
        heapq.heappush(self.events, event)

    def dequeue(self):
        return heapq.heappop(self.events)

    def getMin(self):
        return self.events[0]

class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, event):
        self.queue.append(event)

    def dequeue(self):
        return self.queue.pop(0)

    def size(self):
        return len(self.queue)

class Server:
    """
    A class to represent a server in a queueing system.

    Attributes:
        queue (Queue): The queue of the server.
        distribution (rv_continuous): The service time distribution.
        rng (RandomState): A random number generator.
        server_id (int): The ID of the server.
        mean_service_time (float): The mean service time.
        destination (FlowBranchOperator): The destination after leaving the server.
        in_service (int): The number of customers currently in service.
        total_time_in_service (float): The total time spent in service.
        total_customers_served (int): The total number of customers served.
        max_queue_length (int): The maximum length of the queue.
        reneges (int): The number of customers who reneged.
        total_time_in_queue (float): The total time spent in the queue.
        cumulative_queue_length (int): The cumulative length of the queue.

    Methods:
        get_server_id(): Returns the server ID.
    """
    def __init__(self, distribution, destinations=None, server_id=None):
        """
        Initialize a Server object.

        Args:
            distribution (list): The parameters of the service time distribution.
            destinations (FlowBranchOperator, optional): The destination after leaving the server. Defaults to None.
            server_id (int, optional): The ID of the server. Defaults to None.
        """
        self.queue = Queue()
        self.branch = False
        if distribution[0] == "exponential":
            self.distribution = stats.expon(scale=distribution[1])
        elif distribution[0] == "discrete":
            self.distribution = stats.rv_discrete(values=(distribution[1], distribution[2]))
        elif distribution[0] == "uniform":
            self.distribution = stats.uniform(loc=distribution[1], scale=distribution[2])
        elif distribution[0] == "normal":
            self.distribution = stats.norm(loc=distribution[1], scale=distribution[2])
        elif distribution[0] == "beta":
            self.distribution = stats.beta(a=distribution[1], b=distribution[2], loc=distribution[3], scale=distribution[4])
        elif distribution[0] == "gamma":
            self.distribution = stats.gamma(a=distribution[1], loc=distribution[2], scale=distribution[3])
        elif distribution[0] == "branch":
            self.branch = True
            self.distribution = stats.uniform(loc=0, scale=0)
        elif distribution[0] == "queue":
            self.distribution = None
        else:
            raise ValueError("Distribution not supported")
        self.rng = np.random.RandomState()
        self.server_id = server_id
        if distribution[0] != "branch" and distribution[0] != "queue":
            self.mean_service_time = self.distribution.mean()
        else:
            self.mean_service_time = 0
        self.destination = destinations
        self.in_service = 0
        self.total_time_in_service = 0 
        self.total_customers_served = 0
        self.max_queue_length = 0
        self.reneges = 0
        self.total_time_in_queue = 0
        self.cumulative_queue_length = 0

        # experimental
        self.queue_length_counts = {}
        self.queue_length_times = {}

        self.delayed_departures = 0

    def is_busy(self):
        return self.in_service

    def get_queue_size(self):
        return self.queue.size()

    def is_branch(self):
        return self.branch

    def is_queue(self):
        return self.distribution is None and self.branch == False

    def get_server_id(self):
        return self.server_id

    def get_destination(self):
        return self.destination
        

class Source:
    """
    A class to represent a source in a queueing system.

    Attributes:
        interarrival_time_distribution (rv_continuous): The interarrival time distribution.
        rng (RandomState): A random number generator.
        source_id (int): The ID of the source.
        destination (FlowBranchOperator): The destination after leaving the source.
        total_customers_generated (int): The total number of customers generated.

    Methods:
        get_source_id(): Returns the source ID.
    """
    def __init__(self, distribution, source_id=None, destinations=None):
        """
        Initialize a Source object.

        Args:
            distribution (list): The parameters of the interarrival time distribution.
            destinations (FlowBranchOperator, optional): The destination after leaving the source. Defaults to None.
            source_id (int, optional): The ID of the source. Defaults to None.
        """
        if distribution[0] == "exponential":
            self.distribution = stats.expon(scale=distribution[1])
        elif distribution[0] == "discrete":
            self.distribution = stats.rv_discrete(values=(distribution[1], distribution[2]))
        elif distribution[0] == "uniform":
            self.distribution = stats.uniform(loc=distribution[1], scale=distribution[2])
        elif distribution[0] == "normal":
            self.distribution = stats.norm(loc=distribution[1], scale=distribution[2])
        elif distribution[0] == "beta":
            self.distribution = stats.beta(a=distribution[1], b=distribution[2], loc=distribution[3], scale=distribution[4])
        elif distribution[0] == "gamma":
            self.distribution = stats.gamma(a=distribution[1], loc=distribution[2], scale=distribution[3])
        else:
            raise ValueError("Distribution not supported")
        self.rng = np.random.RandomState()
        self.mean_inter_arrival_time = self.distribution.mean()
        self.destination = destinations
        self.arrival_times = 0
        self.customers_generated = 0
        self.source_id = source_id


class Sim:
    """
    A class to represent a simulation of a queueing system.

    Attributes:
        arrival (int): An identifier for arrival events.
        departure (int): An identifier for departure events.
        generate_log (bool): Whether to generate a log.
        animation (bool): Whether to generate an animation.
        graph_states (list): A list of graph states for the animation.
        seeds (list): A list of seeds for the random number generator.
        num_runs (int): The number of runs of the simulation.
        adj_matrix (np.array): The adjacency matrix of the system.
        queue_list (list): A list of queues.
        forks_probability_matrix (np.array): The probability matrix for the forks.
        distributions (list): A list of distributions for the sources.
        sources (dict): A dictionary of sources.

    Methods:
        __init__(adj_matrix, distributions, queue_list, forks_probability_matrix, seeds=None, num_runs=None, generate_log=False, log_path='logs/', log_name=None, animation=False): Initialize a Sim object.
    """
    arrival = 1
    departure = 2

    def __init__(self, adj_matrix, distributions, queue_list, seeds=None, num_runs=None, generate_log=False, log_path='logs/', log_name=None, animation=False, record_history=False, logging_mode='All'):
        """
        Initialize a Sim object.

        Args:
            adj_matrix (np.array): The adjacency matrix of the system.
            distributions (list): A list of distributions for the sources.
            queue_list (list): A list of queues.
            forks_probability_matrix (np.array): The probability matrix for the forks.
            seeds (list, optional): A list of seeds for the random number generator. Defaults to None.
            num_runs (int, optional): The number of runs of the simulation. Defaults to None.
            generate_log (bool, optional): Whether to generate a log. Defaults to False.
            log_path (str, optional): The path to the log file. Defaults to 'logs/'.
            log_name (str, optional): The name of the log file. Defaults to 'simulation.log'.
            animation (bool, optional): Whether to generate an animation. Defaults to False.
        """
        self.generate_log = generate_log
        self.animation = animation
        self.record_history = record_history
        self.logging_mode = logging_mode

        if self.generate_log:
            if log_name is None:
                filename = log_path + "simulation.log"
            else:
                filename = log_path + log_name
            if os.path.exists(filename):
                # delete old log file
                open(filename, 'w').close()
            # create logger
            logging.basicConfig(filename=filename, filemode='w', level=logging.INFO)
        

        if self.animation:
            self.graph_states = []

        # Set up seeds
        if seeds is not None:
            self.seeds = seeds
            self.num_runs = len(seeds)
        elif num_runs is not None:
            self.seeds = list(range(num_runs))+1000
            self.num_runs = num_runs
        else:
            raise ValueError("Either seeds or num_runs must be provided.")
    

        self.adj_matrix = adj_matrix
        self.queue_list = queue_list
        self.distributions = distributions

        self.sources = {i:Source(distributions[i], source_id=i) for i, source in enumerate(np.diag(adj_matrix)) if source > 0}
        for i, source in self.sources.items():
            destiny = [0 for i in range(len(self.adj_matrix))]
            for j, flow in enumerate(self.adj_matrix[i]):
                if flow > 0 and i != j:
                    destiny[j] = j
            probabilities = adj_matrix[i].copy()
            probabilities[i] = 0
            source.destination = FlowBranchOperator(probabilities=probabilities, children=destiny, origin=i)
            if self.generate_log:
                if self.logging_mode == 'All':
                    logging.info(f"Source {i} has destination {destiny}")
                    logging.info(f"Source {i} has distribution {distributions[i]}")
                    logging.info(f"Source {i} has mean inter-arrival time {source.mean_inter_arrival_time}")

        # change Server(distributions[i]) to Server(abs(server),distributions[i]) for more basic model (to be tested later)
        self.servers = {i:Server(distributions[i], server_id=i) for i, server in enumerate(np.diag(adj_matrix)) if server <= 0}
        for i, server in self.servers.items():
            destiny = [0 for i in range(len(self.adj_matrix))]
            for j, flow in enumerate(self.adj_matrix[i]):
                if flow > 0 and i != j:
                    destiny[j] = j
            probabilities = adj_matrix[i].copy()
            probabilities[i] = 0
            server.destination = FlowBranchOperator(probabilities=probabilities, children=destiny, origin=i)
            if self.generate_log:
                if self.logging_mode == 'All':
                    logging.info(f"Server {i} has destination {destiny}")
                    logging.info(f"Server {i} has distribution {distributions[i]}")
                    logging.info(f"Server {i} has mean service time {server.mean_service_time}")


        # Initialize event list and clock
        self.FutureEventList = EventList()
        self.Clock = 0.0

        # Initialize statistics
        self.total_time_in_queues = 0  
        self.total_customers = 0  
        self.total_reneges = 0  
        self.total_arrival_time = 0 

        self.current_customers_in_system = 0
        self.customers_in_system = [0]

        # Initialize history
        self.avg_queue_length_history = []  # for time spent in queues
        self.avg_server_length_history = []  # for time spent in queues and queue time
        self.total_arrival_time_history = []  
        self.total_service_time_history = []
        self.avg_queue_time_history = []  # for queue time
        self.renege_rate_history = []  # for renege rate
        self.server_utilizations_history = []  # for server utilization
        self.total_customers_history = []
        self.max_queue_lengths_history = []
        self.avg_time_at_server_history = []
        self.customers_served_per_server = []
        self.probabilities_of_queue_lengths_history = []

        # experimental
        self.test_variable = []
        self.test_variable_two = []
        self.test_variable_three = []

    def run(self, number_of_customers=50, use_next_available_server=False):
        """
        Run the simulation.

        Args:
            visualize (bool, optional): Whether to visualize the simulation. Defaults to False.
            number_of_customers (int, optional): The number of customers to simulate. Defaults to 50.
        """
        n = len(np.diag(self.adj_matrix))
        self.server_seeds = [[] for i in range(n)]
        self.source_seeds = [[] for i in range(n)]
        self.number_of_customers = number_of_customers
        self.use_next_available_server = use_next_available_server

        if number_of_customers > 1000:
            if self.logging_mode == 'All':
                logging.info("Animation and logging disabled due to large number of customers")
            self.animation = False
            

        for i, seed in enumerate(self.seeds):
            rng = np.random.RandomState(seed)
            # setup server and source seeds, store them in a list for verification
            for server in self.servers.values():
                server_seed = rng.randint(3, 9999999)
                server.rng = np.random.RandomState(server_seed)
                self.server_seeds[server.server_id].append(server_seed)
            for source in self.sources.values():
                source_seed = rng.randint(3, 9999999)
                source.rng = np.random.RandomState(source_seed)
                self.source_seeds[source.source_id].append(source_seed)

            self.FutureEventList = EventList()
            self.reset_variables()
            self.Initialization()

            self.previous_time = 0 # sim time of previous event

            start_time = time.time()  # real time of simulation start

            # Run simulation
            while self.FutureEventList.events:
                evt = self.FutureEventList.getMin()
                self.FutureEventList.dequeue()


                time_difference = evt.get_time() - self.previous_time
                self.servers[evt.get_server_id()].cumulative_queue_length += time_difference * self.servers[evt.get_server_id()].queue.size()


                for server in self.servers:
                    current_queue_size = self.servers[server].get_queue_size() + self.servers[server].delayed_departures
                    if current_queue_size in self.servers[server].queue_length_times:
                        self.servers[server].queue_length_times[current_queue_size] += time_difference
                    else:
                        self.servers[server].queue_length_times[current_queue_size] = time_difference

            
                self.previous_time = evt.get_time()

                if self.total_customers > number_of_customers-1:
                    break

                self.Clock = evt.get_time()
                if evt.get_type() == self.arrival:
                    self.ProcessArrival(evt)
                else:
                    self.ProcessDeparture(evt)

            end_time = time.time()  # simulation end time
            elapsed_time = end_time - start_time  # calculate the difference
            print(f"{i+1}: {elapsed_time} elapsed time for {self.Clock} simulation time with {self.total_customers} customers")

            if self.generate_log and self.total_customers < 100:
                if self.logging_mode == 'All':
                    logging.info(f"{i+1}: {elapsed_time} elapsed time for {self.Clock} simulation time with {self.total_customers} customers")
            self.calculate_metrics()


        if self.generate_log:
            # close log file
            logging.shutdown()

    def Initialization(self):
        if self.generate_log:
            if self.logging_mode == 'All':
                logging.info(f"Initialization")
                logging.info(f"TIME - EVENT ID - SERVER ID - EVENT TYPE")
        for key, source in self.sources.items():
            time_to_next_arrival = source.distribution.rvs(random_state=source.rng)
            self.total_arrival_time += time_to_next_arrival
            source.arrival_times += time_to_next_arrival
            next_server_id = self.get_destination(key)
            evt = Event(self.arrival, self.Clock + time_to_next_arrival, server_id=next_server_id, source_id=key, event_id=self.total_customers)
            self.total_customers += 1
            source.customers_generated += 1
            self.FutureEventList.enqueue(evt)
            if self.generate_log and self.total_customers < 100:
                if self.logging_mode == 'All':
                    logging.info(f"{self.Clock} - {evt.get_event_id()} - {evt.get_server_id()} - Enqueued arrival at {evt.get_time()}")

    def ProcessArrival(self, evt):
        """
        Note: NOT POSITIVE YET IF THIS IS CORRECT FOR MULTIPLE SOURCES, MULTIPLE SERVERS FROM SAME SOURCE
                                            NEEDS TESTING
        """
        server_id = evt.get_server_id()
        if self.generate_log:
            if self.logging_mode == 'All' and self.total_customers < 100:
                logging.info(f"{self.Clock} - {evt.get_event_id()} - {server_id} - Processing arrival")
            elif self.logging_mode == 'Music':
                logging.info(f"{self.Clock} - {evt.get_event_id()} - {server_id} - arrival")

        # If the target is a server, schedule a departure
        if server_id is not None:
            server = self.servers[server_id]
            # server idle
            if server.in_service == 0:
                self.ScheduleDeparture(server_id, evt.event_id)
            # servyer busy
            else:
                # if queue is not full
                if (server.queue.size() + self.servers[server_id].delayed_departures) < (self.queue_list[server_id]) : 
                    evt.arrival_time = self.Clock
                    server.queue.enqueue(evt)
                    if server.queue.size() > server.max_queue_length:
                        server.max_queue_length = server.queue.size()
                # if queue is full
                else:
                    server.reneges += 1
                    
                    if self.generate_log and self.total_customers < 100:
                        if self.logging_mode == 'All':
                            logging.info(f"{self.Clock} - {evt.get_event_id()} - {server_id} - Customer reneged")


        # If the source is a source, schedule the next arrival
        if evt.get_source_id() is not None:
            self.current_customers_in_system += 1
            source_id = evt.get_source_id()
            source = self.sources[source_id]
            time_to_next_arrival = source.distribution.rvs(random_state=source.rng)
            self.total_arrival_time += time_to_next_arrival
            source.arrival_times += time_to_next_arrival
            source.customers_generated += 1
            evt = Event(self.arrival, self.Clock + time_to_next_arrival, server_id=server_id, source_id=source_id, event_id=self.total_customers)
            self.total_customers += 1
            self.FutureEventList.enqueue(evt)
            if self.generate_log and self.total_customers < 100:
                if self.logging_mode == 'All':
                    logging.info(f"{self.Clock} - {evt.get_event_id()} - {evt.get_server_id()} - Enqueued arrival at {evt.get_time()}")


    def ScheduleDeparture(self, server_id, event_id=None):
        if self.generate_log:
            if self.logging_mode == 'All' and self.total_customers < 100:
                logging.info(f"{self.Clock} - {event_id} - {server_id} - Scheduling departure from server")

        if server_id is not None:
            server = self.servers[server_id]
            server.in_service = 1
            server.total_customers_served += 1
            service_time = 0
            if server.distribution is not None and server.is_branch() == False:
                while service_time <= 0:
                    service_time = server.distribution.rvs(random_state=server.rng)

            if self.generate_log:
                if self.logging_mode == 'Music':
                    logging.info(f"{service_time} - {event_id} - {server_id} - processing")
                    
            server.total_time_in_service += service_time
            departure = Event(self.departure, self.Clock + service_time, server_id=server_id, source_id=None, event_id=event_id)
            self.FutureEventList.enqueue(departure)
            self.FutureEventList.servers_next_departure[server_id] = departure.get_time()


    def ProcessDeparture(self, evt):
        if self.generate_log:
            if self.logging_mode == 'All' and self.total_customers < 100:
                logging.info(f"{self.Clock} - {evt.get_event_id()} - {evt.get_server_id()} - Processing departure from server ")
            elif self.logging_mode == 'Music':
                logging.info(f"{self.Clock} - {evt.get_event_id()} - {evt.get_server_id()} - departure")
        if self.animation:
            self.graph_states.append(self.get_graph_state())
        server_id = evt.get_server_id()
        server = self.servers[server_id]
        if evt.delayed_event:
            server.delayed_departures -= 1
            evt.delayed_event = False
        next_server_id = self.get_destination(server_id)

        # If the current server is a queue, check if any of the next destinations are available
        if next_server_id is None:
            children = server.destination.get_children_ids()
            for child in children:
                if child in self.servers and self.servers[child].is_busy() == 0:
                    next_server_id = child
                    break

        # If current server is a queue with an available destination or a server, schedule a departure
        if next_server_id is not None or server.destination.is_sink():
            
            if server.queue.size() > 0:
                customer = server.queue.dequeue()
                self.total_time_in_queues += self.Clock - customer.get_time()
                server.total_time_in_queue += self.Clock - customer.arrival_time
                self.ScheduleDeparture(server_id, customer.event_id)
                server.in_service = 1
                server.cumulative_queue_length += server.queue.size()

            else:
                server.in_service = 0
                self.FutureEventList.servers_next_departure[server_id] = 0
            if server.destination.is_sink():

                if self.generate_log and self.total_customers < 100:
                    if self.logging_mode == 'All':
                        logging.info(f"{self.Clock} - {evt.get_event_id()} - {evt.get_server_id()} - Customer exited the system")
            else:
                self.ProcessArrival(Event(self.arrival, self.Clock, server_id=next_server_id, source_id=None, event_id=evt.event_id))        
        else:
            # Server is a queue and next destinations are all busy
            # find time of next departure from children servers
            children = server.destination.get_children_ids()
            # before:
            next_departure_times = []
            for child in children:
                next_departure_times.append( self.FutureEventList.get_time_of_next_departure(child))
            next_departure_time = min(next_departure_times)
            # testing with:
            shortest_queue_length = math.inf
            for child in children:
                if child != server_id and self.FutureEventList.get_time_of_next_departure(child) < shortest_queue_length:
                    shortest_queue_length = self.FutureEventList.get_time_of_next_departure(child)
            next_departure_time = shortest_queue_length

            # re-schedule departure
            self.schedule_delayed_departure(server_id, evt.event_id, next_departure_time)
            if self.generate_log and self.total_customers < 100:
                if self.logging_mode == 'All':
                    logging.info(f"{self.Clock} - {evt.get_event_id()} - {evt.get_server_id()} - Customer delayed departure from server")


    def schedule_delayed_departure(self, server_id, event_id, new_departure_time):
        """
        Schedule a departure from the server at the time of the next departure from the children servers
        function used if the server is a queue and the next destinations are all busy
        """
        if self.generate_log and self.total_customers < 100:
            logging.info(f"{self.Clock} - {event_id} - {server_id} - Scheduling delayed departure from server")
        if server_id is not None:
            server = self.servers[server_id]
            server.in_service = 1
            departure = Event(self.departure, new_departure_time, server_id=server_id, source_id=None, event_id=event_id)
            server.delayed_departures += 1
            departure.delayed_event = True
            # not sure about this line
            departure.delayed_time += new_departure_time - self.Clock
            self.FutureEventList.enqueue(departure)
            self.FutureEventList.servers_next_departure[server_id] = departure.get_time()
            # this may not be correct... 
            server.total_time_in_queue += new_departure_time - self.Clock

    def get_destination(self, id):
        """
        Process the destination of a customer.
        If the adjacency matrix contains values not suitable for a probability matrix, the customer will be sent to the shortest queue.
        """
        shortest_queue = False
        if id in self.servers:
            node = self.servers[id]
            if node.is_queue() or node.destination.is_sink():
                return None
            shortest_queue = node.destination.uses_shortest_queue()
        elif id in self.sources:
            node = self.sources[id]
            shortest_queue = node.destination.uses_shortest_queue()
        next_id = None
        if shortest_queue:
            children = node.destination.get_children_ids()
            shortest_queue_length = math.inf
            shortest_queue_id = None
            zero_queue_length = []
            for child in children:
                if child in self.servers and self.servers[child].queue.size() < shortest_queue_length:
                    shortest_queue_length = self.servers[child].queue.size()
                    shortest_queue_id = child
                if child in self.servers and self.servers[child].queue.size() == 0:
                    zero_queue_length.append(child)
            """ 
            *** MAKES A BIG DIFFERENCE IN ALL L, LQ, W, WQ ***
            This commented out code would send customers to the shortest queue with the shortest remaining service time
            Obviously to shorted remaining time is better than the shortest queue,
            but it's not clear if this is what Jaamsim does.
            """
            if self.use_next_available_server:
                shortest_remaining_service_time = math.inf
                for child in zero_queue_length:
                    if self.FutureEventList.get_time_of_next_departure(child) < shortest_remaining_service_time:
                        shortest_remaining_service_time = self.FutureEventList.get_time_of_next_departure(child)
                        shortest_queue_id = child
                

            next_id = shortest_queue_id
        else:
            next_id = node.destination.randomly_select_child()

        return next_id


    def get_queue_lengths(self):
        return [server.queue.size() for server in self.servers]



    # Functions to calculate metrics
    def calculate_metrics(self):
        if self.total_customers == 0:
            return 0, 0, 0, [0] * len(self.servers)
        
        avg_time_at_server = {server : (self.servers[server].total_time_in_service + self.servers[server].total_time_in_queue) / self.servers[server].total_customers_served for server in self.servers if self.servers[server].total_customers_served > 0}
        avg_queue_time = {server : self.servers[server].total_time_in_queue / self.servers[server].total_customers_served for server in self.servers if self.servers[server].total_customers_served > 0}
        
        server_utilizations = {server : self.servers[server].total_time_in_service / self.Clock for server in self.servers}
        max_queue_lengths = {server : self.servers[server].max_queue_length for server in self.servers}
        renege_rate = {server : self.servers[server].reneges / self.servers[server].total_customers_served for server in self.servers if self.servers[server].total_customers_served > 0}

        service_times = {server : self.servers[server].total_time_in_service / self.servers[server].total_customers_served for server in self.servers if self.servers[server].total_customers_served > 0}
        arrival_times = {source : self.sources[source].arrival_times / self.sources[source].customers_generated for source in self.sources}

        customers_served_per_server = {server : self.servers[server].total_customers_served for server in self.servers}

        avg_queue_length = { server: sum([length*time for length, time in self.servers[server].queue_length_times.items()]) / self.Clock for server in self.servers}
        avg_server_length = { server: avg_queue_length[server] + server_utilizations[server] for server in self.servers}

        queue_length_probabilities = { server: {length: time / self.Clock for length, time in self.servers[server].queue_length_times.items()} for server in self.servers}

        # average queue time for for each server with queue_length_times
        #test_variable = self.total_time_in_queues / self.total_customers


        # Store metrics in history
        if self.record_history:
            self.avg_queue_length_history.append(avg_queue_length)
            self.avg_server_length_history.append(avg_server_length)
            self.avg_time_at_server_history.append(avg_time_at_server)
            self.avg_queue_time_history.append(avg_queue_time)
            self.renege_rate_history.append(renege_rate)
            self.server_utilizations_history.append(server_utilizations)
            self.total_arrival_time_history.append(arrival_times)
            self.total_service_time_history.append(service_times)
            self.total_customers_history.append(self.total_customers)
            self.max_queue_lengths_history.append(max_queue_lengths)
            self.customers_served_per_server.append(customers_served_per_server)
            self.probabilities_of_queue_lengths_history.append(queue_length_probabilities)

            # experimental
            #self.test_variable.append(self.total_time_in_queues / self.total_customers)
            #self.test_variable_two.append(test_variable_two)
            #self.test_variable_three.append(test_variable_three)

        total_U = sum(server_utilizations.values())
        total_L = ( sum(avg_queue_length.values()) + sum(server_utilizations.values()) )
        total_LQ = sum(avg_queue_length.values()) 
        total_W = ( sum(avg_time_at_server.values()) + sum(avg_queue_time.values()) )
        total_WQ = sum(avg_queue_time.values()) 
        

        if self.generate_log:
            logging.info(f"Average queue length: {avg_queue_length}")
            logging.info(f"Average server length: {avg_server_length}")
            logging.info(f"Average time at server: {avg_time_at_server}")
            logging.info(f"Average queue_time: {avg_queue_time}")
            logging.info(f"Renege rate: {renege_rate}")
            logging.info(f"Server utilization: {server_utilizations}")
            logging.info(f"Total arrival time: {arrival_times}")
            logging.info(f"Total service time: {service_times}")
            logging.info(f"Total customers served: {self.total_customers}")
            logging.info(f"Max queue length: {max_queue_lengths}")
            logging.info(f"Customers served per server: {customers_served_per_server}")
            #logging.info(f"TEST QUEUE LENGTH: {test_variable}")
            #logging.info(f"TEST QUEUE LENGTH TWO: {test_variable_two}")
            #logging.info(f"TEST QUEUE LENGTH THREE: {test_variable_three}")
            logging.info(f"--------------------------------------------------")
            logging.info(f"Total U: {total_U}")
            logging.info(f"Total L: {total_L}")
            logging.info(f"Total LQ: {total_LQ}")
            logging.info(f"Total W: {total_W}")
            logging.info(f"Total WQ: {total_WQ}")

    def print_metrics(self):
        print(f"Average queue length: {self.avg_queue_length_history}")
        print(f"Average server length: {self.avg_server_length_history}")
        print(f"Average time at server: {self.avg_time_at_server_history}")
        print(f"Average queue_time: {self.avg_queue_time_history}")
        print(f"Renege rate: {self.renege_rate_history}")
        print(f"Server utilization: {self.server_utilizations_history}")
        print(f"Total arrival time: {self.total_arrival_time_history}")
        print(f"Total service time: {self.total_service_time_history}")
        print(f"Total customers served: {self.total_customers_history}")
        print(f"Max queue length: {self.max_queue_lengths_history}")
        print(f"Customers served per server: {self.customers_served_per_server}")

    def print_test_variables(self):
        print(f"TEST QUEUE LENGTH: {self.test_variable}")


    def calculate_confidence_intervals(self, confidence_level):
        queue_time_ci = stats.t.interval(confidence_level, len(self.avg_queue_time_history)-1, loc=np.mean(self.avg_queue_time_history), scale=stats.sem(self.avg_queue_time_history))
        renege_rate_ci = stats.t.interval(confidence_level, len(self.renege_rate_history)-1, loc=np.mean(self.renege_rate_history), scale=stats.sem(self.renege_rate_history))

        return queue_time_ci, renege_rate_ci


    def plot_probability_k_customers_in_system(self, node=1, confidence_level=0.95):
        """
        Plot the probability of k customers in the system for a specific server.

        Args:
            server (int, optional): The ID of the server to plot. Defaults to 1.
            confidence_level (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.

        Raises:
            ValueError: If the specified server is not found.

        Returns:
            None
        """
        
        if node not in self.servers:
            raise ValueError("Server not found")
        
        if self.servers[node].distribution is None:
            rho = 0
            for child in self.servers[node].destination.get_children_ids():
                if child in self.servers:
                    rho += self.server_utilizations_history[-1][child]
            rho = rho / len([child for child in self.servers[node].destination.get_children_ids() if child in self.servers])
            print("Node is a branch or queue and has no distribution so rho is calculated as the average of the server utilizations of the children")
        else:
            rho = [self.server_utilizations_history[i][node] for i in range(len(self.server_utilizations_history))]
            rho = sum(rho) / len(rho)

        print(f"rho = {rho}")
        theoretical_probabilities = []
        for i in range(len(self.servers[node].queue_length_counts)):
            theoretical_probabilities.append(rho ** i * (1 - rho))


        probabilities = [0 for i in range(10)]
        for i in range(len(probabilities)):
                probabilities[i] = sum([self.probabilities_of_queue_lengths_history[j][node][i] for j in range(len(self.probabilities_of_queue_lengths_history))]) / len(self.probabilities_of_queue_lengths_history)

        n = 10
        width = 0.35
        ind = np.arange(n)
        fig, ax = plt.subplots()
        ax.bar(ind - width/2, probabilities[:n], width, label='Simulated')
        ax.bar(ind + width/2, theoretical_probabilities[:n], width, label='Theoretical')

        ax.set_ylabel('Probability')
        ax.set_xlabel('Number of customers in system')
        ax.set_title(f'Probability of k customers in system for server {node}')
        ax.set_xticks(ind)
        ax.set_xticklabels([i for i in range(n)])
        ax.legend()

        plt.show()


    def reset_variables(self):
        self.total_time_in_queues = 0  # for time spent in queues
        self.total_customers = 0
        self.total_reneges = 0
        self.total_arrival_time = 0
        self.Clock = 0.0

        for server in self.servers.values():
            server.queue = Queue()
            server.in_service = 0
            server.total_time_in_service = 0
            server.total_customers_served = 0
            server.max_queue_length = 0
            server.reneges = 0
            server.total_time_in_queue = 0
            server.cumulative_queue_length = 0
   
            server.queue_length_counts = {}
            server.queue_length_times = {}
            server.queue_length_times[0] = 0

            server.delayed_departures = 0


        for source in self.sources.values():
            source.arrival_times = 0
            source.customers_generated = 0


    def plot_metrics(self, server=1, confidence_level=0.95):
        """
        Plot the metrics of a specific server.

        Args:
            server (int, optional): The ID of the server to plot. Defaults to 1.
            confidence_level (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.

        Raises:
            ValueError: If the specified server is not found.

        Returns:
            None
        """
        if self.record_history == False:
            print("No history recorded")
            return None

        if server not in self.servers:
            raise ValueError("Server not found")

        plt.figure(figsize=(24, 24))
        plt.suptitle(f"Server {server}")

        # Plot average server length
        avg_server_length = [x[server] for x in self.avg_server_length_history]
        plt.subplot(4, 2, 1)
        if len(avg_server_length) == 1:
            plt.plot(avg_server_length*np.ones(2))
        else:
            plt.plot(avg_server_length)
            try:
                avg_server_length_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(avg_server_length), scale=stats.sem(avg_server_length))
                plt.plot([avg_server_length_ci[0]] * len(avg_server_length), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([avg_server_length_ci[1]] * len(avg_server_length), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        plt.xlabel("Simulation run")
        plt.ylabel("Average server length")
        plt.title(f"Average server length with {confidence_level*100}% confidence interval")

        # Plot average queue length
        avg_queue_length = [x[server] for x in self.avg_queue_length_history]
        plt.subplot(4, 2, 2)
        if len(avg_queue_length) == 1:
            plt.plot(avg_queue_length*np.ones(2))
        else:
            plt.plot(avg_queue_length)
            try:
                avg_queue_length_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(avg_queue_length), scale=stats.sem(avg_queue_length))
                plt.plot([avg_queue_length_ci[0]] * len(avg_queue_length), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([avg_queue_length_ci[1]] * len(avg_queue_length), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        plt.xlabel("Simulation run")
        plt.ylabel("Average queue length")
        plt.title(f"Average queue length with {confidence_level*100}% confidence interval")


        # Plot average time at server
        avg_time_at_server = [x[server] for x in self.avg_time_at_server_history]
        plt.subplot(4, 2, 3)
        if len(avg_time_at_server) == 1:
            plt.plot(avg_time_at_server*np.ones(2))
        else:
            plt.plot(avg_time_at_server)
            try:
                avg_time_at_server_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(avg_time_at_server), scale=stats.sem(avg_time_at_server))
                plt.plot([avg_time_at_server_ci[0]] * len(avg_time_at_server), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([avg_time_at_server_ci[1]] * len(avg_time_at_server), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass

        plt.xlabel("Simulation run")
        plt.ylabel("Average time at server")
        plt.title(f"Average time at server with {confidence_level*100}% confidence interval")

        # Plot queue time
        queue_time = [x[server] for x in self.avg_queue_time_history]
        plt.subplot(4, 2, 4)
        if len(queue_time) == 1:
            plt.plot(queue_time*np.ones(2))
        else:
            plt.plot(queue_time)
            try: 
                queue_time_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(queue_time), scale=stats.sem(queue_time))
                plt.plot([queue_time_ci[0]] * len(queue_time), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([queue_time_ci[1]] * len(queue_time), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        plt.legend()
        plt.xlabel("Simulation run")
        plt.ylabel("Queue time")
        plt.title(f"Queue time with {confidence_level*100}% confidence interval")

        # Plot server utilization
        server_utilizations = [x[server] for x in self.server_utilizations_history]
        plt.subplot(4, 2, 5)
        if len(server_utilizations) == 1:
            plt.plot(server_utilizations*np.ones(2))
        else:
            plt.plot(server_utilizations)
            try:
                server_utilizations_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(server_utilizations), scale=stats.sem(server_utilizations))
                plt.plot([server_utilizations_ci[0]] * len(server_utilizations), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([server_utilizations_ci[1]] * len(server_utilizations), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        plt.legend()
        plt.xlabel("Simulation run")
        plt.ylabel("Server utilization")
        plt.title(f"Server utilization with {confidence_level*100}% confidence interval")

        # Plot renege rate
        renege_rate = [x[server] for x in self.renege_rate_history]
        plt.subplot(4, 2, 6)
        if len(renege_rate) == 1:
            plt.plot(renege_rate)
        else:
            plt.plot(renege_rate)
            try: 
                renege_rate_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(renege_rate), scale=stats.sem(renege_rate))
                plt.plot([renege_rate_ci[0]] * len(renege_rate), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([renege_rate_ci[1]] * len(renege_rate), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        plt.legend()
        plt.xlabel("Simulation run")
        plt.ylabel("Renege rate")
        plt.title(f"Renege rate with {confidence_level*100}% confidence interval")

        # Plot average service time
        avg_service_time = [x[server] for x in self.total_service_time_history]
        plt.subplot(4, 2, 7)
        if len(avg_service_time) == 1:
            plt.plot(avg_service_time*np.ones(2))
        else:
            plt.plot(avg_service_time)
            try:
                avg_service_time_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(avg_service_time), scale=stats.sem(avg_service_time))
                plt.plot([avg_service_time_ci[0]] * len(avg_service_time), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([avg_service_time_ci[1]] * len(avg_service_time), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        plt.legend()
        plt.xlabel("Simulation run")
        plt.ylabel("Average service time")
        plt.title(f"Average service time with {confidence_level*100}% confidence interval")

        # Plot average arrival time
        # calculate all sources connected to server
        sources = []
        for i, source in enumerate(np.diag(self.adj_matrix)):
            if source > 0:
                sources.append(i)
        connected_sources = []
        for source in sources:
            for i,node in enumerate(self.adj_matrix[source]):
                if i == server and node == 1:
                    connected_sources.append(source)

        avg_arrival_time = [x[source] for x in self.total_arrival_time_history for source in connected_sources]
        plt.subplot(4, 2, 8)
        if len(avg_arrival_time) == 1:
            plt.plot(avg_arrival_time*np.ones(2))
        else:
            plt.plot(avg_arrival_time)
            try:
                avg_arrival_time_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(avg_arrival_time), scale=stats.sem(avg_arrival_time))
                plt.plot([avg_arrival_time_ci[0]] * len(avg_arrival_time), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([avg_arrival_time_ci[1]] * len(avg_arrival_time), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        plt.legend()
        plt.xlabel("Simulation run")
        plt.ylabel("Average arrival time")
        plt.title(f"Average arrival time with {confidence_level*100}% confidence interval")


        plt.show()

    def plot_metrics_all_servers(self, confidence_level=0.95, predicted={}, theoretical={}):
        """
        Plot the overall metrics of the system.
        note: -intended for single X/X/c system checks, not for X/X/c/c systems or larger networks of servers/queues
              -function is very rough and needs to be cleaned up... not good for general use

        Args:
            confidence_level (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.
            predicted (dict, optional): A dictionary of predicted values for the metrics. Defaults to {}. L, LQ, W, WQ keys required.

        Returns:
            None
        """
        if self.record_history == False:
            print("No history recorded")
            return None


        plt.figure(figsize=(24, 24))
        plt.suptitle(f"Metrics for all servers")

        # Plot average queue length
        avg_cumulative_queue_length = [sum([x[server] for server in self.servers.keys()]) for x in self.avg_queue_length_history]
        plt.subplot(2, 2, 1)
        plt.xticks(np.arange(0, len(avg_cumulative_queue_length), 1.0))
        if len(avg_cumulative_queue_length) == 1:
            plt.plot(avg_cumulative_queue_length*np.ones(2))
        else:
            plt.plot(avg_cumulative_queue_length)
            try:
                avg_queue_length_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(avg_cumulative_queue_length), scale=stats.sem(avg_cumulative_queue_length))
                plt.plot([avg_queue_length_ci[0]] * len(avg_cumulative_queue_length), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([avg_queue_length_ci[1]] * len(avg_cumulative_queue_length), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        if 'LQ' in predicted:
            if len(predicted['LQ']) == 1:
                plt.plot(predicted['LQ']*np.ones(len(avg_cumulative_queue_length)), label='Predicted')
            else:
                plt.plot(predicted['LQ'], label='Predicted')
        if 'LQ' in theoretical:
            if len(theoretical['LQ']) == 1:
                plt.plot(theoretical['LQ']*np.ones(len(avg_cumulative_queue_length)), label='Theoretical')
            else:
                plt.plot(theoretical['LQ'], label='Theoretical')
        plt.legend(prop={'size': 14})
        plt.xlabel("Simulation run")
        plt.ylabel("Average queue length")
        plt.title(f"Average queue length with {confidence_level*100}% confidence interval")

        # Plot average server length
        avg_cumulative_server_length = [sum([x[server] for server in self.servers.keys()]) for x in self.avg_server_length_history]
        
        
        plt.subplot(2, 2, 2)
        plt.xticks(np.arange(0, len(avg_cumulative_server_length), 1.0))
        if len(avg_cumulative_server_length) == 1:
            plt.plot(avg_cumulative_server_length*np.ones(2))
        else:
            plt.plot(avg_cumulative_server_length)
            try:
                avg_server_length_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(avg_cumulative_server_length), scale=stats.sem(avg_cumulative_server_length))
                plt.plot([avg_server_length_ci[0]] * len(avg_cumulative_server_length), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([avg_server_length_ci[1]] * len(avg_cumulative_server_length), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        if 'L' in predicted:
            if len(predicted['L']) == 1:
                plt.plot(predicted['L']*np.ones(len(avg_cumulative_server_length)), label='Predicted')
            else:
                plt.plot(predicted['L'], label='Predicted')
        if 'L' in theoretical:
            if len(theoretical['L']) == 1:
                plt.plot(theoretical['L']*np.ones(len(avg_cumulative_server_length)), label='Theoretical')
            else:
                plt.plot(theoretical['L'], label='Theoretical')
        plt.legend(prop={'size': 14})
        plt.xlabel("Simulation run")
        plt.ylabel("Average server length")
        plt.title(f"Average server length with {confidence_level*100}% confidence interval")



        # Plot queue time
        avg_cumulative_queue_time = [sum([x[server] for server in self.servers.keys()]) for x in self.avg_queue_time_history]
        try:
            # Plot average time at server
            cumulative_queues = [sum([x[server] for server in self.servers.keys() if self.servers[server].is_queue()]) for x in self.avg_queue_time_history]
            cumulative_servers = [sum([x[server] for server in self.servers.keys() if not self.servers[server].is_queue()]) for x in self.avg_queue_time_history]
            #print(cumulative_servers)
            #print(len(self.servers)-len([1 for server in self.servers.keys() if self.servers[server].is_queue() or self.servers[server].is_branch()]))
            # NOTE: THIS IS TERRIBLE. IN FUTURE SHOULD REDUCE NUMBER OF CALCULATIONS
            cumulative_servers = [x/(len(self.servers)-len([1 for server in self.servers.keys() if self.servers[server].is_queue() or self.servers[server].is_branch()])) for x in cumulative_servers]
            avg_cumulative_queue_time = [cumulative_queues[i] + (cumulative_servers[i]) for i in range(len(cumulative_queues))]
        except:
            # no queues in system
            avg_cumulative_queue_time = [sum([x[server] for server in self.servers.keys()]) for x in self.avg_queue_time_history]
        plt.subplot(2, 2, 3)
        plt.xticks(np.arange(0, len(avg_cumulative_queue_time), 1.0))
        if len(avg_cumulative_queue_time) == 1:
            plt.plot(avg_cumulative_queue_time*np.ones(2))
        else:
            plt.plot(avg_cumulative_queue_time)
            try: 
                queue_time_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(avg_cumulative_queue_time), scale=stats.sem(avg_cumulative_queue_time))
                plt.plot([queue_time_ci[0]] * len(avg_cumulative_queue_time), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([queue_time_ci[1]] * len(avg_cumulative_queue_time), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        if 'WQ' in predicted:
            if len(predicted['WQ']) == 1:
                plt.plot(predicted['WQ']*np.ones(len(avg_cumulative_queue_time)), label='Predicted')
            else:
                plt.plot(predicted['WQ'], label='Predicted')
        if 'WQ' in theoretical:
            if len(theoretical['WQ']) == 1:
                plt.plot(theoretical['WQ']*np.ones(len(avg_cumulative_queue_time)), label='Theoretical')
            else:
                plt.plot(theoretical['WQ'], label='Theoretical')
        plt.legend(prop={'size': 14})
        plt.xlabel("Simulation run")
        plt.ylabel("Queue time")
        plt.title(f"Queue time with {confidence_level*100}% confidence interval")


        try:
            # Plot average time at server
            cumulative_queues = [sum([x[server] for server in self.servers.keys() if self.servers[server].is_queue()]) for x in self.avg_time_at_server_history]
            cumulative_servers = [sum([x[server] for server in self.servers.keys() if not self.servers[server].is_queue()]) for x in self.avg_time_at_server_history]
            # NOTE: THIS IS TERRIBLE. IN FUTURE SHOULD ADD METRICS TO REDUCE NUMBER OF CALCULATIONS
            cumulative_servers = [x/(len(self.servers)-len([1 for server in self.servers.keys() if self.servers[server].is_queue() or self.servers[server].is_branch()])) for x in cumulative_servers]
            avg_cumulative_time_at_server = [cumulative_queues[i] + (cumulative_servers[i]) for i in range(len(cumulative_queues))]
        except:
            # no queues in system
            avg_cumulative_time_at_server = [sum([x[server] for server in self.servers]) for x in self.avg_time_at_server_history]
        plt.subplot(2, 2, 4)
        plt.xticks(np.arange(0, len(avg_cumulative_time_at_server), 1.0))
        if len(avg_cumulative_time_at_server) == 1:
            plt.plot(avg_cumulative_time_at_server*np.ones(2))
        else:
            plt.plot(avg_cumulative_time_at_server)
            try:
                avg_time_at_server_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(avg_cumulative_time_at_server), scale=stats.sem(avg_cumulative_time_at_server))
                plt.plot([avg_time_at_server_ci[0]] * len(avg_cumulative_time_at_server), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([avg_time_at_server_ci[1]] * len(avg_cumulative_time_at_server), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        if 'W' in predicted:
            if len(predicted['W']) == 1:
                plt.plot(predicted['W']*np.ones(len(avg_cumulative_time_at_server)), label='Predicted')
            else:
                plt.plot(predicted['W'], label='Predicted')
        if 'W' in theoretical:
            if len(theoretical['W']) == 1:
                plt.plot(theoretical['W']*np.ones(len(avg_cumulative_time_at_server)), label='Theoretical')
            else:
                plt.plot(theoretical['W'], label='Theoretical')
        plt.legend(prop={'size': 14})
        plt.xlabel("Simulation run")
        plt.ylabel("Average time at server")
        plt.title(f"Average time at server with {confidence_level*100}% confidence interval")


        plt.show()

    def plot_cumulative_renege_rates(self, queue_length=None, confidence_level=0.95, observed=None):
        """
        Plot the combined cumulative renege rates of all servers.
        # note: only works for 1 source connected to all servers
        * checks if all servers exponential, and assumes average rate between them if so
        * if not all servers exponential, uses a variance weighted average of the rates
        * Note very reusable... I was just trying to get something working in time for specific problem

        Args:
            confidence_level (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.

        Returns:
            None
        """

        renege_rates = [sum([x[server] for server in self.servers.keys()]) for x in self.renege_rate_history]
        
        if self.record_history == False:
            print("No history recorded")
            return 0

        if queue_length is not None:
            avg_lam = 0
            server_count = 0
            is_exponential = True
            for i, server in self.servers.items():
                if server.distribution != None and self.distributions[i][0] != 'exponential':
                    is_exponential = False
                    break
                elif server.distribution != None:
                    avg_lam += self.distributions[i][1]
                    server_count += 1
            avg_lam = avg_lam / server_count

            avg_mu = 0
            source_count = 0
            for i, source in self.sources.items():
                avg_mu += self.distributions[i][1]
                source_count += 1
            avg_mu = avg_mu / source_count

            theo = calculate_theoretical_renege_rate(avg_lam, avg_mu, server_count, queue_length)
            if not is_exponential:
                theo = theo*(1-(avg_lam/avg_mu)**server_count)

        plt.figure(figsize=(12, 8))
        plt.xticks(np.arange(0, len(renege_rates), 1.0))
        if len(renege_rates) == 1:
            plt.plot(renege_rates*np.ones(2))
        else:
            plt.plot(renege_rates)
            try:
                renege_rate_ci = stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean(renege_rates), scale=stats.sem(renege_rates))
                plt.plot([renege_rate_ci[0]] * len(renege_rates), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
                plt.plot([renege_rate_ci[1]] * len(renege_rates), 'r--', alpha=.5, label=f"{confidence_level*100}% confidence interval")
            except:
                pass
        if observed is not None:
            if len(renege_rates) == 1:
                plt.plot(observed*np.ones(2), label='Observed')
            else:
                plt.plot(observed*np.ones(len(renege_rates)), label='Observed')
        if queue_length is not None:
            if len(renege_rates) == 1:
                plt.plot(theo*np.ones(2), label='Theoretical')
            else:
                plt.plot(theo*np.ones(len(renege_rates)), label='Theoretical')
        plt.legend()
        plt.xlabel("Simulation run")
        plt.ylabel("Cumulative renege rate")
        plt.title(f"Cumulative renege rate with {confidence_level*100}% confidence interval")

        plt.show()


# create a class to test and analyze the simulation model and plot the results
        # Test over a range of server utilizations

class SimTester:
    """
    A class to test and analyze the simulation model and plot the results.

    # Note: it's only set up to accuractly test mu=1, with varrying lambda

    * Note: Only works with 1 of the tests below at a time. Cannot run two tests then plot each individually.
    -> test_server_utilization(utilization lists) - changes only 1 server's utilization at a time
        -> use plot_metrics(server) to plot the metrics of the server
    -> test_all_servers(distribution settings) - changes all servers' distributions at once
        -> use plot_metrics_all_servers() to plot the metrics of all servers

    Attributes:
        adj_matrix (np.array): The adjacency matrix of the system.
        distributions (list): A list of distributions for the sources.
        queue_list (list): A list of queues.
        forks_probability_matrix (np.array): The probability matrix for the forks.
        seeds (list, optional): A list of seeds for the random number generator. Defaults to None.
        num_runs (int, optional): The number of runs of the simulation. Defaults to None.
        generate_log (bool, optional): Whether to generate a log. Defaults to False.
        sources (dict): A dictionary of sources.
        servers (dict): A dictionary of servers.
        queue_times (dict): A dictionary of Queue times.
        renege_rates (dict): A dictionary of renege rates.
        observed_utilizations (dict): A dictionary of observed utilizations.
        avg_queue_length_history (dict): A dictionary of average queue lengths.

    Methods:
        __init__(adj_matrix, distributions, queue_list, forks_probability_matrix, seeds=None, num_runs=None, generate_log=False): Initialize a SimTester object.
        test_server_utilization(server_utilizations, number_of_customers=50): Test the simulation over a range of server utilizations.
    """
    def __init__(self, adj_matrix, distributions, queue_list, seeds=None, num_runs=None, generate_log=False, record_history=False):
        """
        Initialize a SimTester object. Note: Does not run the simulation until a test is called.

        Args:
            adj_matrix (np.array): The adjacency matrix of the system.
            distributions (list): A list of distributions for the sources.
            queue_list (list): A list of queues.
            forks_probability_matrix (np.array): The probability matrix for the forks.
            seeds (list, optional): A list of seeds for the random number generator. Defaults to None.
            num_runs (int, optional): The number of runs of the simulation. Defaults to None.
            generate_log (bool, optional): Whether to generate a log. Defaults to False.
        """
        self.adj_matrix = adj_matrix
        self.distributions = distributions
        self.queue_list = queue_list
        self.seeds = seeds
        self.num_runs = num_runs
        self.generate_log = generate_log
        self.record_history = record_history

        self.sources = None
        self.servers = None

        self.queue_times = {}
        self.renege_rates = {}
        self.observed_utilizations = {}
        self.avg_queue_length_history = {}

    def test_server_utilization(self, server_utilizations, number_of_customers=50):
        """
        Test the simulation over a range of server utilizations.

        Args:
            server_utilizations (list): A list of server utilizations to test.
            number_of_customers (int, optional): The number of customers to simulate. Defaults to 50.
        """
        self.test_server_utilizations = server_utilizations
        count = 0
        for i, server_utilization in enumerate(server_utilizations):
            if server_utilization != []:
                for utilization in server_utilization:
                    if self.generate_log:
                        print(f"Testing server utilization {server_utilization}")
                    distributions = self.distributions
                    distributions[i][1] = utilization
                    sim = Sim(self.adj_matrix, distributions, self.queue_list, seeds=self.seeds, num_runs=self.num_runs, generate_log=self.generate_log, record_history=self.record_history)
                    sim.run(number_of_customers=number_of_customers)
                    if count == 0:
                        self.server_seeds = sim.server_seeds
                        self.source_seeds = sim.source_seeds
                        self.sources = sim.sources
                        self.servers = sim.servers
                    count += 1

                    avg_queue_length = np.mean([x[i] for x in sim.avg_queue_length_history])
                    aq_sem = stats.sem([x[i] for x in sim.avg_queue_length_history])
                    if i not in self.avg_queue_length_history:
                        self.avg_queue_length_history[i] = [(avg_queue_length, aq_sem)]
                    else:
                        self.avg_queue_length_history[i].append((avg_queue_length, aq_sem))

                    queue_time = np.mean([x[i] for x in sim.avg_queue_time_history])
                    rt_sem = stats.sem([x[i] for x in sim.avg_queue_time_history])
                    if i not in self.queue_times:
                        self.queue_times[i] = [(queue_time, rt_sem)]
                    else:
                        self.queue_times[i].append((queue_time, rt_sem))

                    renege_rate = np.mean([x[i] for x in sim.renege_rate_history])
                    rr_sem = stats.sem([x[i] for x in sim.renege_rate_history])
                    if i not in self.renege_rates:
                        self.renege_rates[i] = [(renege_rate, rr_sem)]
                    else:
                        self.renege_rates[i].append((renege_rate, rr_sem))

                    observed_utilization = np.mean([x[i] for x in sim.server_utilizations_history])
                    ou_sem = stats.sem([x[i] for x in sim.server_utilizations_history])
                    if i not in self.observed_utilizations:
                        self.observed_utilizations[i] = [(observed_utilization, ou_sem)]
                    else:
                        self.observed_utilizations[i].append((observed_utilization, ou_sem))
                        
        return self.queue_times, self.renege_rates
    

    def plot_metrics(self, server=1, confidence_level=0.95):
        """
        Plot the metrics of a specific server.

        This method generates four plots: average queue length, average queue time, renege rate, and server utilization, 
        each as a function of server utilization. Each plot includes a confidence interval.

        Args:
            server (int, optional): The ID of the server to plot. Defaults to 1.
            confidence_level (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.

        Raises:
            ValueError: If the specified server is not found.

        Returns:
            None
        """

 
        queue_lengths = self.avg_queue_length_history[server]
        queue_times = self.queue_times[server]
        renege_rates = self.renege_rates[server]
        server_utilizations = self.observed_utilizations[server]

        queue_lengths_ci = [stats.t.interval(confidence_level, len(self.seeds)-1, loc=aql_mean, scale=aql_sem) for aql_mean, aql_sem in queue_lengths]
        queue_time_ci = [stats.t.interval(confidence_level, len(self.seeds)-1, loc=rt_mean, scale=rt_sem) for rt_mean, rt_sem in queue_times]
        renege_rate_ci = [stats.t.interval(confidence_level, len(self.seeds)-1, loc=rr_mean, scale=rr_sem) for rr_mean, rr_sem in renege_rates]
        server_utilization_ci = [stats.t.interval(confidence_level, len(self.seeds)-1, loc=ou_mean, scale=ou_sem) for ou_mean, ou_sem in server_utilizations]

        exp_servers = 1
       # Check if all servers exponential
        if not all([dist[0] == "exponential" for dist in self.distributions]):
            print("Not all servers are exponential, theoretical values not calculated")
            exp_servers = 0
        else:
            server_dist_type = self.distributions[server][0]
            if server_dist_type == "deterministic":
                mu = self.distributions[server][1]
                LQ = [ (util*util) / (2*(1-util)) for util in self.test_server_utilizations[server]]
                WQ = [ util / (2*(1-util)) for util in self.test_server_utilizations[server]]
            elif server_dist_type == "exponential":
                LQ = [( util*util ) / (1-util) for util in self.test_server_utilizations[server]]
                WQ = [ (util / ((1/util)*(1-util)) ) for util in self.test_server_utilizations[server]]
            else:
                print("Distribution not supported")
                exp_servers = 0


        plt.figure(figsize=(20, 20))
        plt.suptitle(f"Server {server}")
        plt.subplot(2, 2, 1)
        plt.xticks(range(len(server_utilizations)), ['%.2f' % util for util in self.test_server_utilizations[server]])
        plt.plot([aql_mean for aql_mean, aql_sem in queue_lengths], label="Average queue length", color='b', linestyle='--', linewidth=6)
        plt.plot([aql_ci[0] for aql_ci in queue_lengths_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='lightsteelblue', linestyle='--', linewidth=6)
        plt.plot([aql_ci[1] for aql_ci in queue_lengths_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='mediumpurple', linestyle='--', linewidth=6)
        if exp_servers:
            plt.plot(LQ, label="Theoretical average queue length", color='r', linewidth=9, linestyle=(0, (1, 10)))
        plt.legend(prop={'size': 12})
        plt.xlabel("Server utilization")
        plt.ylabel("Average queue length")
        plt.title(f"Average queue length with {confidence_level*100}% confidence interval")

        plt.subplot(2, 2, 2)
        plt.xticks(range(len(server_utilizations)), ['%.2f' % util for util in self.test_server_utilizations[server]])
        plt.plot([rt_mean for rt_mean, rt_sem in queue_times], label="Queue times", color='b', linestyle='--', linewidth=6)
        plt.plot([rt_ci[0] for rt_ci in queue_time_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='lightsteelblue', linestyle='--', linewidth=6)
        plt.plot([rt_ci[1] for rt_ci in queue_time_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='mediumpurple', linestyle='--', linewidth=6)
        if exp_servers:
            plt.plot(WQ, label="Theoretical queue time", color='r', linewidth=9,linestyle=(0, (1, 10)))
        plt.legend(prop={'size': 12})
        plt.xlabel("Server utilization")
        plt.ylabel("Average queue time")
        plt.title(f"Average queue time with {confidence_level*100}% confidence interval")

        plt.subplot(2, 2, 3)
        plt.xticks(range(len(server_utilizations)), ['%.2f' % util for util in self.test_server_utilizations[server]])
        plt.plot([rr_mean for rr_mean, rr_sem in renege_rates], label="Renege rate", color='b', linestyle='--', linewidth=5)
        plt.plot([rr_ci[0] for rr_ci in renege_rate_ci], alpha=0.5, label=f"{confidence_level*100}% confidence interval", color='lightsteelblue', linestyle='--', linewidth=6)
        plt.plot([rr_ci[1] for rr_ci in renege_rate_ci], alpha=0.5, label=f"{confidence_level*100}% confidence interval", color='mediumpurple', linestyle='--', linewidth=6)
        plt.legend(prop={'size': 12})
        plt.xlabel("Server utilization")
        plt.ylabel("Renege rate")
        plt.title(f"Renege rate with {confidence_level*100}% confidence interval")

        plt.subplot(2, 2, 4)
        plt.xticks(range(len(server_utilizations)), ['%.2f' % util for util in self.test_server_utilizations[server]])
        plt.plot([ou_mean for ou_mean, ou_sem in server_utilizations], label="Server utilization", color='b', linestyle='--', linewidth=6)
        plt.plot([ou_ci[0] for ou_ci in server_utilization_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", linestyle='--', linewidth=6)
        plt.plot([ou_ci[1] for ou_ci in server_utilization_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", linestyle='--', linewidth=6)
        plt.legend(prop={'size': 12})
        plt.xlabel("Server utilization")
        plt.ylabel("Server utilization")
        plt.title(f"Server utilization with {confidence_level*100}% confidence interval")
        plt.fill_between(range(len(server_utilizations)), [ou_ci[0] for ou_ci in server_utilization_ci], [ou_ci[1] for ou_ci in server_utilization_ci], color='b', alpha=.1)

        plt.show()


    # analyze the important nodes in the network
    def find_principle_servers(self, confidence_level=0.95):
        """
        Analyze the important nodes in the network.

        This method finds the servers with the highest average queue length, average queue time, renege rate, and server utilization.

        Args:
            confidence_level (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.

        Returns:
            tuple: A tuple containing sorted lists of average queue lengths, queue times, renege rates, and server utilizations, 
            along with their corresponding confidence intervals.
        """
        # find the servers with the highest average queue length
        avg_queue_lengths = {server : np.mean([x[server] for x in self.avg_queue_length_history]) for server in self.avg_queue_length_history}
        avg_queue_lengths_ci = {server : stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean([x[server] for x in self.avg_queue_length_history]), scale=stats.sem([x[server] for x in self.avg_queue_length_history])) for server in self.avg_queue_length_history}

        avg_queue_lengths_sorted = sorted(avg_queue_lengths.items(), key=lambda x: x[1], reverse=True)
        avg_queue_lengths_ci_sorted = sorted(avg_queue_lengths_ci.items(), key=lambda x: x[1][0], reverse=True)

        # find the servers with the highest average queue time
        queue_times = {server : np.mean([x[server] for x in self.queue_times]) for server in self.queue_times}
        queue_times_ci = {server : stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean([x[server] for x in self.queue_times]), scale=stats.sem([x[server] for x in self.queue_times])) for server in self.queue_times}

        queue_times_sorted = sorted(self.queue_times.items(), key=lambda x: x[1], reverse=True)
        queue_times_ci_sorted = sorted(queue_times_ci.items(), key=lambda x: x[1][0], reverse=True)

        # find the servers with the highest renege rate
        renege_rates = {server : np.mean([x[server] for x in self.renege_rates]) for server in self.renege_rates}
        renege_rates_ci = {server : stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean([x[server] for x in self.renege_rates]), scale=stats.sem([x[server] for x in self.renege_rates])) for server in self.renege_rates}

        renege_rates_sorted = sorted(renege_rates.items(), key=lambda x: x[1], reverse=True)
        renege_rates_ci_sorted = sorted(renege_rates_ci.items(), key=lambda x: x[1][0], reverse=True)

        # find the servers with the highest server utilization
        server_utilizations = {server : np.mean([x[server] for x in self.observed_utilizations]) for server in self.observed_utilizations}
        server_utilizations_ci = {server : stats.t.interval(confidence_level, len(self.seeds)-1, loc=np.mean([x[server] for x in self.observed_utilizations]), scale=stats.sem([x[server] for x in self.observed_utilizations])) for server in self.observed_utilizations}

        server_utilizations_sorted = sorted(server_utilizations.items(), key=lambda x: x[1], reverse=True)
        server_utilizations_ci_sorted = sorted(server_utilizations_ci.items(), key=lambda x: x[1][0], reverse=True)

        return avg_queue_lengths_sorted, avg_queue_lengths_ci_sorted, queue_times_sorted, queue_times_ci_sorted, renege_rates_sorted, renege_rates_ci_sorted, server_utilizations_sorted, server_utilizations_ci_sorted
    

    # display information about the important nodes in the network
    def display_principle_servers(self, confidence_level=0.95):
        """
        Display information about the important nodes in the network.

        This method prints the servers with the highest average queue length, average queue time, renege rate, and server utilization, 
        along with their corresponding confidence intervals.

        Args:
            confidence_level (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.
        """
        avg_queue_lengths_sorted, avg_queue_lengths_ci_sorted, queue_times_sorted, queue_times_ci_sorted, renege_rates_sorted, renege_rates_ci_sorted, server_utilizations_sorted, server_utilizations_ci_sorted = self.find_principle_servers(confidence_level)

        print(f"Average queue lengths: {avg_queue_lengths_sorted}")
        print(f"Average queue lengths confidence intervals: {avg_queue_lengths_ci_sorted}")
        print(f"Average queue times: {queue_times_sorted}")
        print(f"Average queue times confidence intervals: {queue_times_ci_sorted}")
        print(f"Renege rates: {renege_rates_sorted}")
        print(f"Renege rates confidence intervals: {renege_rates_ci_sorted}")
        print(f"Server utilizations: {server_utilizations_sorted}")
        print(f"Server utilizations confidence intervals: {server_utilizations_ci_sorted}")


    # analyze the seeds used in the simulation
    def randomness_check(self, confidence_level=0.95, sample_size=10000):
        """
        Analyze the seeds used in the simulation.

        This method checks the randomness of the seeds by generating a large sample of random numbers and comparing their mean and variance 
        to the expected mean and variance.

        Plot 1 shows the chi-square statistic for each seed used in the servers, and plot 2 shows the chi-square statistic for each seed used in the sources.

        Args:
            confidence_level (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.
            sample_size (int, optional): The size of the sample to generate. Defaults to 10000.
        """
        length = 0
        chi_square_servers = []

        for j,seed_list in enumerate(self.server_seeds):
            if len(set(seed_list)) != len(seed_list):
                print("Warning: seeds are not unique")
                break
            if seed_list == []:
                continue
            chi_square_servers.append([])
            chi_square_servers[-1].append(j)
            for i, seed in enumerate(seed_list):
                if seed < 0 or seed > 9999999:
                    print("Warning: seeds are not between 0 and 9999999")
                    break
                rng = np.random.RandomState(seed)
                sample = rng.uniform(size=sample_size)
                sample = np.histogram(sample, bins=int(math.ceil(math.sqrt(sample_size))))[0]
                chi_square, p_value = stats.chisquare(sample)
                chi_square_servers[-1].append(chi_square)
                if i > length:
                    length = i


        chi_square_sources = []
        for j,seed_list in enumerate(self.source_seeds):
            if len(set(seed_list)) != len(seed_list):
                print("Warning: seeds are not unique")
                break
            chi_square_sources.append([])
            chi_square_sources[-1].append(j)
            for i, seed in enumerate(seed_list):
                if seed < 0 or seed > 9999999:
                    print("Warning: seeds are not between 0 and 9999999")
                    break
                rng = np.random.RandomState(seed)
                sample = rng.uniform(size=sample_size)
                sample = np.histogram(sample, bins=int(math.ceil(math.sqrt(sample_size))))[0]
                chi_square, p_value = stats.chisquare(sample)
                chi_square_sources[-1].append(chi_square)

        reject_limit = stats.chi2.ppf(confidence_level, int(math.ceil(math.sqrt(sample_size)))-1)

        plt.figure(figsize=(20, 20))
        plt.subplot(2, 1, 1)
        plt.plot(reject_limit * np.ones(length), 'r--', label=f"{confidence_level*100}% confidence interval")
        for i, chi_square in enumerate(chi_square_servers):
            if chi_square_servers[i] != []:
                plt.plot(chi_square_servers[i][1:], label=f"Server {chi_square_servers[i][0]}")
        plt.xlabel("Seed")
        plt.ylabel("Chi-square statistic")
        plt.title("Chi-square statistic for seeds used in servers")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(reject_limit * np.ones(length), 'r--', label=f"{confidence_level*100}% confidence interval")
        for i, chi_square in enumerate(chi_square_sources):
            if chi_square_sources[i] != []:
                plt.plot(chi_square_sources[i][1:], label=f"Source {chi_square_sources[i][0]}")
        plt.xlabel("Source")
        plt.ylabel("Chi-square statistic")
        plt.title("Chi-square statistic for seeds used in sources")
        plt.legend()

        plt.show()


    def test_all_servers(self, distribution_settings, number_of_customers=1000):
        """
        Test the simulation over a range of server distribution settings.
        **** INCOMPLETE AND NOT WORKING ****

        
        Args:
            distribution_settings (list): A list of distribution settings to test.
            number_of_customers (int, optional): The number of customers to simulate. Defaults to 1000.
        """
        self.test_distribution_settings = distribution_settings
        count = 0
        for distribution_setting in distribution_settings:
            for i, server_setting in enumerate(distribution_setting):
                if self.generate_log:
                    print(f"Testing server setting {server_setting}")
                distributions = self.distributions
                for i in np.diag(self.adj_matrix):
                    if i == -1:
                        distributions[i] = server_setting
            sim = Sim(self.adj_matrix, distributions, self.queue_list, seeds=self.seeds, num_runs=self.num_runs, generate_log=self.generate_log)
            sim.run(number_of_customers=number_of_customers)
            if count == 0:
                self.server_seeds = sim.server_seeds
                self.source_seeds = sim.source_seeds
                self.sources = sim.sources
                self.servers = sim.servers
            count += 1

            avg_cumulative_queue_lengths = np.mean([sum([x[server] for server in self.servers.keys()]) for x in sim.avg_queue_length_history])
            aql_sem = stats.sem([sum([x[server] for server in self.servers.keys()]) for x in sim.avg_queue_length_history])
            if i not in self.avg_queue_length_history:
                self.avg_queue_length_history[i] = [(avg_cumulative_queue_lengths, aql_sem)]
            else:
                self.avg_queue_length_history[i].append((avg_cumulative_queue_lengths, aql_sem))

            avg_cumulative_queue_times = np.mean([sum([x[server] for server in self.servers.keys()]) for x in sim.avg_queue_time_history])
            rt_sem = stats.sem([sum([x[server] for server in self.servers.keys()]) for x in sim.avg_queue_time_history])
            if i not in self.queue_times:
                self.queue_times[i] = [(avg_cumulative_queue_times, rt_sem)]
            else:
                self.queue_times[i].append((avg_cumulative_queue_times, rt_sem))

            avg_cumulative_renege_rates = np.mean([sum([x[server] for server in self.servers.keys()]) for x in sim.renege_rate_history])
            rr_sem = stats.sem([sum([x[server] for server in self.servers.keys()]) for x in sim.renege_rate_history])
            if i not in self.renege_rates:
                self.renege_rates[i] = [(avg_cumulative_renege_rates, rr_sem)]
            else:
                self.renege_rates[i].append((avg_cumulative_renege_rates, rr_sem))

            avg_cumulative_server_utilizations = np.mean([sum([x[server] for server in self.servers.keys()]) for x in sim.server_utilizations_history])
            ou_sem = stats.sem([sum([x[server] for server in self.servers.keys()]) for x in sim.server_utilizations_history])
            if i not in self.observed_utilizations:
                self.observed_utilizations[i] = [(avg_cumulative_server_utilizations, ou_sem)]
            else:
                self.observed_utilizations[i].append((avg_cumulative_server_utilizations, ou_sem))

                
        return self.queue_times, self.renege_rates
    

    def plot_metrics_all_servers(self, confidence_level=0.95, predictions=None, theoretical_values=None):
        """
        Plot the metrics of all servers.

        This method generates four plots: average queue length, average queue time, renege rate, and server utilization, 
        each as a function of server distribution settings. Each plot includes a confidence interval.

        Args:
            confidence_level (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.

        Raises:
            ValueError: If the specified server is not found.

        Returns:
            None
        """

        queue_lengths = self.avg_queue_length_history
        queue_times = self.queue_times
        renege_rates = self.renege_rates
        server_utilizations = self.observed_utilizations

        queue_lengths_ci = [stats.t.interval(confidence_level, len(self.seeds)-1, loc=aql_mean, scale=aql_sem) for aql_mean, aql_sem in queue_lengths]
        queue_time_ci = [stats.t.interval(confidence_level, len(self.seeds)-1, loc=rt_mean, scale=rt_sem) for rt_mean, rt_sem in queue_times]
        renege_rate_ci = [stats.t.interval(confidence_level, len(self.seeds)-1, loc=rr_mean, scale=rr_sem) for rr_mean, rr_sem in renege_rates]
        server_utilization_ci = [stats.t.interval(confidence_level, len(self.seeds)-1, loc=ou_mean, scale=ou_sem) for ou_mean, ou_sem in server_utilizations]

        plt.figure(figsize=(20, 20))
        plt.suptitle(f"Metrics for all servers")

        plt.subplot(2, 2, 1)
        plt.xticks(range(len(self.test_distribution_settings)), [str(setting) for setting in self.test_distribution_settings])
        plt.plot([aql_mean for aql_mean, aql_sem in queue_lengths], label="Average queue length", color='b', linestyle='--')
        plt.plot([aql_ci[0] for aql_ci in queue_lengths_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='lightsteelblue', linestyle='--')
        plt.plot([aql_ci[1] for aql_ci in queue_lengths_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='mediumpurple', linestyle='--')
        if theoretical_values:
            try:
                plt.plot(theoretical_values['LQ'], label="Theoretical average queue length", color='r', linewidth=1.5, linestyle=(0, (1, 10)))
            except:
                pass
        plt.legend()
        plt.xlabel("Server distribution settings")
        plt.ylabel("Average queue length")
        plt.title(f"Average queue length with {confidence_level*100}% confidence interval")

        plt.subplot(2, 2, 2)
        plt.xticks(range(len(self.test_distribution_settings)), [str(setting) for setting in self.test_distribution_settings])
        plt.plot([rt_mean for rt_mean, rt_sem in queue_times], label="Queue times", color='b', linestyle='--')
        plt.plot([rt_ci[0] for rt_ci in queue_time_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='lightsteelblue', linestyle='--')
        plt.plot([rt_ci[1] for rt_ci in queue_time_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='mediumpurple', linestyle='--')
        if theoretical_values:
            try:
                plt.plot(theoretical_values['WQ'], label="Theoretical queue time", color='r', linewidth=1.5,linestyle=(0, (1, 10)))
            except:
                pass    
        plt.legend()
        plt.xlabel("Server distribution settings")
        plt.ylabel("Average queue time")
        plt.title(f"Average queue time with {confidence_level*100}% confidence interval")

        plt.subplot(2, 2, 3)
        plt.xticks(range(len(self.test_distribution_settings)), [str(setting) for setting in self.test_distribution_settings])
        plt.plot([rr_mean for rr_mean, rr_sem in renege_rates], label="Renege rate", color='b', linestyle='--')
        plt.plot([rr_ci[0] for rr_ci in renege_rate_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='lightsteelblue', linestyle='--')
        plt.plot([rr_ci[1] for rr_ci in renege_rate_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", color='mediumpurple', linestyle='--')
        if theoretical_values:
            try:
                plt.plot(theoretical_values['RR'], label="Theoretical renege rate", color='r', linewidth=1.5,linestyle=(0, (1, 10)))
            except:
                pass
        plt.legend()
        plt.xlabel("Server distribution settings")
        plt.ylabel("Renege rate")
        plt.title(f"Renege rate with {confidence_level*100}% confidence interval")

        plt.subplot(2, 2, 4)
        plt.xticks(range(len(self.test_distribution_settings)), [str(setting) for setting in self.test_distribution_settings])
        plt.plot([ou_mean for ou_mean, ou_sem in server_utilizations])
        plt.plot([ou_ci[0] for ou_ci in server_utilization_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", linestyle='--')
        plt.plot([ou_ci[1] for ou_ci in server_utilization_ci], alpha=.5, label=f"{confidence_level*100}% confidence interval", linestyle='--')
        if theoretical_values:
            try:
                plt.plot(theoretical_values['utilization'], label="Theoretical server utilization", color='r', linewidth=1.5,linestyle=(0, (1, 10)))
            except:
                pass
        plt.legend()
        plt.xlabel("Server distribution settings")
        plt.ylabel("Server utilization")
        plt.title(f"Server utilization with {confidence_level*100}% confidence interval")
        plt.fill_between(range(len(self.test_distribution_settings)), [ou_ci[0] for ou_ci in server_utilization_ci], [ou_ci[1] for ou_ci in server_utilization_ci], color='b', alpha=.1)

        plt.show()


def calculate_theoretical_renege_rate(lam, mu, c, N):
    N = N + c

    rho = lam / (c * mu)
    a = lam / mu

    p_zero = 1 / (1 + sum([(a**(n)) / math.factorial(n) for n in range(1,c+1)]) + ((a**c) / math.factorial(c)) * sum([(rho ** (n-c)) for n in range(c+1, N+1)]))
    p_n =  ((a**(N)) * p_zero)  /  ( math.factorial(c) * c**(N-c) )

    return p_n


class Analyze_Simulation:
    """
    INCOMPLETE DUE TO TIME CONSTRAINTS


    A class to determine if the simulation matches a set of theoretical values from a given system or data.

    Attributes:
        adj_sim (np.array): The adjacency matrix of the system.
        distributions_sim (list): A list of distributions for the sources.
        queue_list_sim (list): A list of queue capacities.
        theoretical_values (optional): A dictionary of theoretical values for the system. Defaults to None.
        data (optional): A dictionary of data for the system. Defaults to None.
    
    Methods:
        __init__(adj_sim, distributions_sim, queue_list_sim, theoretical_values=None, data=None): Initialize an Analyze_Simulation object.
        analyze(): Analyze the simulation.

    """
    def __init__(self, adj_sim, distributions_sim, queue_list_sim, theoretical_values=None, data=None):
        """
        Initialize an Analyze_Simulation object.

        Args:
            adj_sim (np.array): The adjacency matrix of the system.
            distributions_sim (list): A list of distributions for the sources.
            queue_list_sim (list): A list of queue capacities.
            theoretical_values (optional): A dictionary of theoretical values for the system. Defaults to None.
            data (optional): A dictionary of data for the system. Defaults to None.
        """
        self.adj_sim = adj_sim
        self.distributions_sim = distributions_sim
        self.queue_list_sim = queue_list_sim
        self.theoretical_values = theoretical_values
        self.data = data

    def analyze(self):
        # NOTE: THIS IS WRONG
        """
        Analyze the simulation.

        This method compares the theoretical values to the simulation values and determines if the simulation matches the theoretical values.

        Returns:
            float: The p-value of the simulation.
        """

        return 0