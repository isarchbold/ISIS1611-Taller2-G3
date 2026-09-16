from abc import ABC, abstractmethod

from algorithms.evaluation import evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1) y cada
        acción consume un ply. Debe respetar el orden de las acciones legales,
        usar evaluation_function en terminales y cortes, y contar cada estado
        procesado una vez en self.nodes_evaluated, incluida la raíz.

        Tips:
        - Use state.get_legal_actions(agent_index) y
          state.generate_successor(agent_index, action) para expandir el árbol.
        - Compruebe state.is_win(), state.is_lose() y el corte de profundidad;
          evalúe esos estados con evaluation_function(state).
        - El siguiente agente es (agent_index + 1) % state.get_num_agents().
          depth=1 incluye una acción de MAX y depth=2 una de MAX y una de MIN.
        - Reinicie las métricas y cuente una vez cada estado procesado, incluida
          la raíz. Retorne la acción de MAX y conserve la primera en los empates.
        """
        # TODO: Add your code here
        self.nodes_evaluated = 0
        def minimax_value(
            current_state: GameState,
            agent_index: int,
            current_depth: int,
        ) -> float:
            self.nodes_evaluated += 1
            if (
                current_state.is_win()
                or current_state.is_lose()
                or current_depth >= self.depth
            ):
                return evaluation_function(current_state)

            legal_actions = current_state.get_legal_actions(agent_index)
            if not legal_actions:
                return evaluation_function(current_state)

            next_agent = (
                agent_index + 1
            ) % current_state.get_num_agents()
            if agent_index == 0:
                best_value = float("-inf")
                for action in legal_actions:
                    successor = current_state.generate_successor(
                        agent_index, action
                    )
                    value = minimax_value(
                        successor,
                        next_agent,
                        current_depth + 1,
                    )
                    best_value = max(best_value, value)

                return best_value
            
            best_value = float("inf")
            for action in legal_actions:
                successor = current_state.generate_successor(
                    agent_index, action
                )
                value = minimax_value(
                    successor,
                    next_agent,
                    current_depth + 1,
                )
                best_value = min(best_value, value)
            return best_value

        self.nodes_evaluated += 1
        if state.is_win() or state.is_lose():
            return None

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        best_action = legal_actions[0]
        best_value = float("-inf")
        next_agent = 1 % state.get_num_agents()
        for action in legal_actions:
            successor = state.generate_successor(0, action)
            value = minimax_value(successor, next_agent, 1)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        # TODO: Add your code here
        self.nodes_evaluated = 0

        def buscar(estado, agente, profundidad, alfa, beta):
            self.nodes_evaluated += 1

            if (estado.is_win() or estado.is_lose() or profundidad == 0):
                
                return evaluation_function(estado), None

            acciones = estado.get_legal_actions(agente)

            if not acciones:
                return evaluation_function(estado), None

            siguiente = (agente + 1) % estado.get_num_agents()
            mejor_accion = acciones[0]

            if agente == 0:
                mejor_valor = float("-inf")

                for accion in acciones:
                    nuevo_estado = estado.generate_successor(agente, accion)

                    valor, _ = buscar(
                        nuevo_estado,
                        siguiente,
                        profundidad - 1,
                        alfa,
                        beta,
                    )

                    if valor > mejor_valor:
                        mejor_valor = valor
                        mejor_accion = accion

                    alfa = max(alfa, mejor_valor)

                    if mejor_valor >= beta:
                        break

            else:
                mejor_valor = float("inf")

                for accion in acciones:
                    nuevo_estado = estado.generate_successor(agente, accion)

                    valor, _ = buscar(nuevo_estado, siguiente, profundidad - 1, alfa, beta)

                    if valor < mejor_valor:
                        mejor_valor = valor
                        mejor_accion = accion

                    beta = min(beta, mejor_valor)

                    if mejor_valor <= alfa:
                        break

            return mejor_valor, mejor_accion

        _, mejor_accion = buscar(state, 0, self.depth, float("-inf"), float("inf"))

        return mejor_accion

