from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time

EXPECTIMAX_ACTION_WEIGHTS = {
    "move north": 4,
    "charge": 4,
}

TIE_BREAKING_ORDER = [
    "drop off",
    "pick up",
    "charge",
    "move north",
    "move east",
    "move south",
    "move west",
    "park",
]



#helper func
def _task_value(env: WarehouseEnv, robot):
    """
    Estimate how useful the current package situation is for one robot.
    """
    # If the robot already carries a package, estimate the value of finishing it
    if robot.package is not None:
        reward = 3 * manhattan_distance(
            robot.package.position,
            robot.package.destination
        )
        distance_to_destination = manhattan_distance(
            robot.position,
            robot.package.destination
        )
        return reward - distance_to_destination

    # Otherwise, choose the best active package on the board
    best_value = 0

    for package in env.packages:
        if not package.on_board:
            continue

        reward = 3 * manhattan_distance(package.position, package.destination)
        distance_to_package = manhattan_distance(robot.position, package.position)
        delivery_distance = manhattan_distance(package.position, package.destination)

        value = reward - distance_to_package - delivery_distance

        if value > best_value:
            best_value = value

    return best_value


# TODO: section a : 3
#selen implemetation
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    opponent_id = (robot_id + 1) % 2
    robot = env.get_robot(robot_id)
    opponent = env.get_robot(opponent_id)

    def active_packages():
        # return [package for package in env.packages[0:2] if package.on_board]
        return [package for package in env.packages if package.on_board]

    def nearest_charge_distance(curr_robot):
        if len(env.charge_stations) == 0:
            return 0
        return min(
            manhattan_distance(curr_robot.position, station.position)
            for station in env.charge_stations
        )


    def battery_penalty(curr_robot):
        # Estimate how much battery is needed for the next useful task
        if curr_robot.package is not None:
            needed = manhattan_distance(curr_robot.position, curr_robot.package.destination)
        else:
            needed = 0
            for package in active_packages():
                candidate = (
                    manhattan_distance(curr_robot.position, package.position)
                    + manhattan_distance(package.position, package.destination)
                )

                if needed == 0 or candidate < needed:
                    needed = candidate

        shortage = max(0, needed - curr_robot.battery)
        low_battery_charge_distance = (
            nearest_charge_distance(curr_robot)
            if curr_robot.battery <= 5
            else 0
        )

        return 5 * shortage + low_battery_charge_distance

    credit_difference = robot.credit - opponent.credit
    task_difference = _task_value(env, robot) - _task_value(env, opponent)
    battery_difference = robot.battery - opponent.battery
    penalty_difference = battery_penalty(robot) - battery_penalty(opponent)

    return (
        10 * credit_difference
        + 2 * task_difference
        + 0.5 * battery_difference
        - penalty_difference
    )



def _operator_rank(operator):
    if operator in TIE_BREAKING_ORDER:
        return TIE_BREAKING_ORDER.index(operator)
    return len(TIE_BREAKING_ORDER)


def _ordered_operators(operators):
    return sorted(operators, key=_operator_rank)


def _next_robot(robot_id):
    return (robot_id + 1) % 2


def _minimax_value(env, root_robot_id, current_robot_id, depth, heuristic_fn):
    if depth <= 0 or env.done():
        return heuristic_fn(env, root_robot_id)

    operators = _ordered_operators(env.get_legal_operators(current_robot_id))
    if len(operators) == 0:
        return heuristic_fn(env, root_robot_id)

    next_robot_id = _next_robot(current_robot_id)

    if current_robot_id == root_robot_id:
        best_value = float("-inf")

        for operator in operators:
            child = env.clone()
            child.apply_operator(current_robot_id, operator)

            value = _minimax_value(
                child,
                root_robot_id,
                next_robot_id,
                depth - 1,
                heuristic_fn
            )

            if value > best_value:
                best_value = value

        return best_value

    best_value = float("inf")

    for operator in operators:
        child = env.clone()
        child.apply_operator(current_robot_id, operator)

        value = _minimax_value(
            child,
            root_robot_id,
            next_robot_id,
            depth - 1,
            heuristic_fn
        )

        if value < best_value:
            best_value = value

    return best_value


class _SearchTimeout(Exception):
    pass


def _minimax_value_with_deadline(env, root_robot_id, current_robot_id, depth, heuristic_fn, deadline):
    if time.time() >= deadline:
        raise _SearchTimeout()

    if depth <= 0 or env.done():
        return heuristic_fn(env, root_robot_id)

    operators = _ordered_operators(env.get_legal_operators(current_robot_id))
    if len(operators) == 0:
        return heuristic_fn(env, root_robot_id)

    next_robot_id = _next_robot(current_robot_id)

    if current_robot_id == root_robot_id:
        best_value = float("-inf")

        for operator in operators:
            if time.time() >= deadline:
                raise _SearchTimeout()

            child = env.clone()
            child.apply_operator(current_robot_id, operator)

            value = _minimax_value_with_deadline(
                child,
                root_robot_id,
                next_robot_id,
                depth - 1,
                heuristic_fn,
                deadline
            )

            if value > best_value:
                best_value = value

        return best_value

    best_value = float("inf")

    for operator in operators:
        if time.time() >= deadline:
            raise _SearchTimeout()

        child = env.clone()
        child.apply_operator(current_robot_id, operator)

        value = _minimax_value_with_deadline(
            child,
            root_robot_id,
            next_robot_id,
            depth - 1,
            heuristic_fn,
            deadline
        )

        if value < best_value:
            best_value = value

    return best_value


def _minimax_decision_with_deadline(env, robot_id, depth, heuristic_fn, deadline):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if len(operators) == 0:
        return "park"

    best_operator = operators[0]
    best_value = float("-inf")
    opponent_id = _next_robot(robot_id)

    for operator in operators:
        if time.time() >= deadline:
            raise _SearchTimeout()

        child = env.clone()
        child.apply_operator(robot_id, operator)

        value = _minimax_value_with_deadline(
            child,
            robot_id,
            opponent_id,
            depth - 1,
            heuristic_fn,
            deadline
        )

        if value > best_value:
            best_value = value
            best_operator = operator

    return best_operator







# TODO: section b : fixed-depth helper for deterministic grading
def minimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited minimax.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    if depth <= 0:
        depth = 1

    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if len(operators) == 0:
        return "park"

    best_operator = operators[0]
    best_value = float("-inf")
    opponent_id = _next_robot(robot_id)

    for operator in operators:
        child = env.clone()
        child.apply_operator(robot_id, operator)

        value = _minimax_value(
            child,
            robot_id,
            opponent_id,
            depth - 1,
            heuristic_fn
        )

        if value > best_value:
            best_value = value
            best_operator = operator

    return best_operator



def _alphabeta_value(env, root_robot_id, current_robot_id, depth, alpha, beta, heuristic_fn):
    if depth <= 0 or env.done():
        return heuristic_fn(env, root_robot_id)

    operators = _ordered_operators(env.get_legal_operators(current_robot_id))
    if len(operators) == 0:
        return heuristic_fn(env, root_robot_id)

    next_robot_id = _next_robot(current_robot_id)

    if current_robot_id == root_robot_id:
        value = float("-inf")

        for operator in operators:
            child = env.clone()
            child.apply_operator(current_robot_id, operator)

            child_value = _alphabeta_value(
                child,
                root_robot_id,
                next_robot_id,
                depth - 1,
                alpha,
                beta,
                heuristic_fn
            )

            if child_value > value:
                value = child_value

            if value > alpha:
                alpha = value

            if alpha >= beta:
                break

        return value

    value = float("inf")

    for operator in operators:
        child = env.clone()
        child.apply_operator(current_robot_id, operator)

        child_value = _alphabeta_value(
            child,
            root_robot_id,
            next_robot_id,
            depth - 1,
            alpha,
            beta,
            heuristic_fn
        )

        if child_value < value:
            value = child_value

        if value < beta:
            beta = value

        if alpha >= beta:
            break

    return value


def _alphabeta_value_with_deadline(
    env,
    root_robot_id,
    current_robot_id,
    depth,
    alpha,
    beta,
    heuristic_fn,
    deadline
):
    if time.time() >= deadline:
        raise _SearchTimeout()

    if depth <= 0 or env.done():
        return heuristic_fn(env, root_robot_id)

    operators = _ordered_operators(env.get_legal_operators(current_robot_id))
    if len(operators) == 0:
        return heuristic_fn(env, root_robot_id)

    next_robot_id = _next_robot(current_robot_id)

    if current_robot_id == root_robot_id:
        value = float("-inf")

        for operator in operators:
            if time.time() >= deadline:
                raise _SearchTimeout()

            child = env.clone()
            child.apply_operator(current_robot_id, operator)

            child_value = _alphabeta_value_with_deadline(
                child,
                root_robot_id,
                next_robot_id,
                depth - 1,
                alpha,
                beta,
                heuristic_fn,
                deadline
            )

            if child_value > value:
                value = child_value

            if value > alpha:
                alpha = value

            if alpha >= beta:
                break

        return value

    value = float("inf")

    for operator in operators:
        if time.time() >= deadline:
            raise _SearchTimeout()

        child = env.clone()
        child.apply_operator(current_robot_id, operator)

        child_value = _alphabeta_value_with_deadline(
            child,
            root_robot_id,
            next_robot_id,
            depth - 1,
            alpha,
            beta,
            heuristic_fn,
            deadline
        )

        if child_value < value:
            value = child_value

        if value < beta:
            beta = value

        if alpha >= beta:
            break

    return value


def _alphabeta_decision_with_deadline(env, robot_id, depth, heuristic_fn, deadline):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if len(operators) == 0:
        return "park"

    best_operator = operators[0]
    best_value = float("-inf")
    alpha = float("-inf")
    beta = float("inf")
    opponent_id = _next_robot(robot_id)

    for operator in operators:
        if time.time() >= deadline:
            raise _SearchTimeout()

        child = env.clone()
        child.apply_operator(robot_id, operator)

        value = _alphabeta_value_with_deadline(
            child,
            robot_id,
            opponent_id,
            depth - 1,
            alpha,
            beta,
            heuristic_fn,
            deadline
        )

        if value > best_value:
            best_value = value
            best_operator = operator

        if best_value > alpha:
            alpha = best_value

    return best_operator






# TODO: section c : fixed-depth helper for deterministic grading
def alphabeta_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited alpha-beta pruning.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    if depth <= 0:
        depth = 1

    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if len(operators) == 0:
        return "park"

    best_operator = operators[0]
    best_value = float("-inf")
    alpha = float("-inf")
    beta = float("inf")
    opponent_id = _next_robot(robot_id)

    for operator in operators:
        child = env.clone()
        child.apply_operator(robot_id, operator)

        value = _alphabeta_value(
            child,
            robot_id,
            opponent_id,
            depth - 1,
            alpha,
            beta,
            heuristic_fn
        )

        if value > best_value:
            best_value = value
            best_operator = operator

        if best_value > alpha:
            alpha = best_value

    return best_operator



def _expectimax_action_weight(operator):
    if operator in EXPECTIMAX_ACTION_WEIGHTS:
        return EXPECTIMAX_ACTION_WEIGHTS[operator]
    return 1


def _expectimax_value(env, root_robot_id, current_robot_id, depth, heuristic_fn):
    if depth <= 0 or env.done():
        return heuristic_fn(env, root_robot_id)

    operators = _ordered_operators(env.get_legal_operators(current_robot_id))
    if len(operators) == 0:
        return heuristic_fn(env, root_robot_id)

    next_robot_id = _next_robot(current_robot_id)

    # Our robot: choose the maximum expected value.
    if current_robot_id == root_robot_id:
        best_value = float("-inf")

        for operator in operators:
            child = env.clone()
            child.apply_operator(current_robot_id, operator)

            value = _expectimax_value(
                child,
                root_robot_id,
                next_robot_id,
                depth - 1,
                heuristic_fn
            )

            if value > best_value:
                best_value = value

        return best_value

    # opponent turn: use the given action weights
    total_weight = 0
    weighted_sum = 0

    for operator in operators:
        weight = _expectimax_action_weight(operator)
        total_weight += weight

        child = env.clone()
        child.apply_operator(current_robot_id, operator)

        value = _expectimax_value(
            child,
            root_robot_id,
            next_robot_id,
            depth - 1,
            heuristic_fn
        )

        weighted_sum += weight * value

    return weighted_sum / total_weight


def _expectimax_value_with_deadline(env, root_robot_id, current_robot_id, depth, heuristic_fn, deadline):
    if time.time() >= deadline:
        raise _SearchTimeout()

    if depth <= 0 or env.done():
        return heuristic_fn(env, root_robot_id)

    operators = _ordered_operators(env.get_legal_operators(current_robot_id))
    if len(operators) == 0:
        return heuristic_fn(env, root_robot_id)

    next_robot_id = _next_robot(current_robot_id)

    if current_robot_id == root_robot_id:
        best_value = float("-inf")

        for operator in operators:
            if time.time() >= deadline:
                raise _SearchTimeout()

            child = env.clone()
            child.apply_operator(current_robot_id, operator)

            value = _expectimax_value_with_deadline(
                child,
                root_robot_id,
                next_robot_id,
                depth - 1,
                heuristic_fn,
                deadline
            )

            if value > best_value:
                best_value = value

        return best_value

    total_weight = 0
    weighted_sum = 0

    for operator in operators:
        if time.time() >= deadline:
            raise _SearchTimeout()

        weight = _expectimax_action_weight(operator)
        total_weight += weight

        child = env.clone()
        child.apply_operator(current_robot_id, operator)

        value = _expectimax_value_with_deadline(
            child,
            root_robot_id,
            next_robot_id,
            depth - 1,
            heuristic_fn,
            deadline
        )

        weighted_sum += weight * value

    return weighted_sum / total_weight


def _expectimax_decision_with_deadline(env, robot_id, depth, heuristic_fn, deadline):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if len(operators) == 0:
        return "park"

    best_operator = operators[0]
    best_value = float("-inf")
    opponent_id = _next_robot(robot_id)

    for operator in operators:
        if time.time() >= deadline:
            raise _SearchTimeout()

        child = env.clone()
        child.apply_operator(robot_id, operator)

        value = _expectimax_value_with_deadline(
            child,
            robot_id,
            opponent_id,
            depth - 1,
            heuristic_fn,
            deadline
        )

        if value > best_value:
            best_value = value
            best_operator = operator

    return best_operator







# TODO: section d : fixed-depth helper for deterministic grading
def expectimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited expectimax.
    The opponent's legal actions are weighted by EXPECTIMAX_ACTION_WEIGHTS;
    every legal action not in the dictionary has weight 1.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic

    if depth <= 0:
        depth = 1

    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if len(operators) == 0:
        return "park"

    best_operator = operators[0]
    best_value = float("-inf")
    opponent_id = _next_robot(robot_id)

    for operator in operators:
        child = env.clone()
        child.apply_operator(robot_id, operator)

        value = _expectimax_value(
            child,
            robot_id,
            opponent_id,
            depth - 1,
            heuristic_fn
        )

        if value > best_value:
            best_value = value
            best_operator = operator

    return best_operator





class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = _ordered_operators(env.get_legal_operators(agent_id))
        if len(operators) == 0:
            return "park"

        best_operator = operators[0]

        safety_margin = 0.02
        deadline = time.time() + max(0.001, time_limit - safety_margin)

        depth = 1
        while time.time() < deadline:
            try:
                current_best = _minimax_decision_with_deadline(
                    env,
                    agent_id,
                    depth,
                    smart_heuristic,
                    deadline
                )

                best_operator = current_best
                depth += 1

            except _SearchTimeout:
                break

        return best_operator


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = _ordered_operators(env.get_legal_operators(agent_id))
        if len(operators) == 0:
            return "park"

        best_operator = operators[0]

        # leave a small time margin
        safety_margin = 0.02
        deadline = time.time() + max(0.001, time_limit - safety_margin)

        depth = 1

        while time.time() < deadline:
            try:
                current_best = _alphabeta_decision_with_deadline(
                    env,
                    agent_id,
                    depth,
                    smart_heuristic,
                    deadline
                )

                best_operator = current_best
                depth += 1

            except _SearchTimeout:
                break

        return best_operator
    
    
    
    
class AgentExpectimax(Agent):
    # TODO: section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = _ordered_operators(env.get_legal_operators(agent_id))
        if len(operators) == 0:
            return "park"

        best_operator = operators[0]

        safety_margin = 0.02
        deadline = time.time() + max(0.001, time_limit - safety_margin)

        depth = 1

        while time.time() < deadline:
            try:
                current_best = _expectimax_decision_with_deadline(
                    env,
                    agent_id,
                    depth,
                    smart_heuristic,
                    deadline
                )

                best_operator = current_best
                depth += 1

            except _SearchTimeout:
                break

        return best_operator
    
    
    

class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check if a move is illegal the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
