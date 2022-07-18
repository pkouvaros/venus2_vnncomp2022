"""
# File: os_sip.py
# Top contributors (to current version):
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: One Step Symbolic Interval Propagation.
"""

from venus.network.node import *
from venus.bounds.bounds import Bounds
from venus.bounds.os_sip import OSSIP
from venus.bounds.equation import Equation
from venus.common.logger import get_logger
from venus.common.configuration import Config

torch.set_num_threads(1)

class BSSIP:

    logger = None

    def __init__(self, prob, config: Config):
        """
        Arguments:

            prob:
                The verification problem.
            config:
                Configuration.
        """
        self.prob = prob
        self.config = config
        if BSSIP.logger is None and config.LOGGER.LOGFILE is not None:
            BSSIP.logger = get_logger(__name__, config.LOGGER.LOGFILE)
 

    def set_bounds(
        self,
        node: Node,
        lower_slopes: dict=None,
        upper_slopes: dict=None,
        concretisations: bool=True,
        os_sip: OSSIP=None
    ) -> Bounds:
        in_flag = self._get_stability_flag(node.from_node[0])
        out_flag = self._get_out_prop_flag(node)
        symb_eq = Equation.derive(
            node,
            self.config,
            None if out_flag is None else out_flag.flatten(),
            None if in_flag is None else in_flag.flatten(),
        )

        lower_bounds, upper_bounds, bs_flag = self.back_substitution(
            symb_eq, node, out_flag, lower_slopes, upper_slopes, os_sip
        )

        if len(lower_bounds) == 0:
            return

        lower_bounds = torch.max(
            node.bounds.lower[bs_flag].flatten(), lower_bounds
        )
        upper_bounds = torch.min(
            node.bounds.upper[bs_flag].flatten(), upper_bounds
        )
        
        self._set_bounds(
            node, Bounds(lower_bounds, upper_bounds), lower_slopes, upper_slopes, bs_flag
        )

        if node.has_relu_activation() is True:
            return node.to_node[0].get_unstable_count()

        return 0

    def _set_bounds(
        self,
        node: None,
        bounds: Bounds,
        lower_slopes: torch.Tensor=None,
        upper_slopes: torch.Tensor=None,
        out_flag: torch.tensor=None
    ):
        if node.has_relu_activation() and \
        lower_slopes is not None and \
        upper_slopes is not None:
            # relu node with custom slope - leave slope as is but remove from
            # it newly stable nodes.
            old_fl = node.to_node[0].get_unstable_flag() 
            new_fl = out_flag[old_fl]

            lower_slopes[node.to_node[0].id] = lower_slopes[node.to_node[0].id][new_fl]
            upper_slopes[node.to_node[0].id] = upper_slopes[node.to_node[0].id][new_fl]

        node.update_bounds(bounds, out_flag)

    def _get_out_prop_flag(self, node: Node):
        if node.has_relu_activation():
            return node.to_node[0].get_unstable_flag()

        elif len(node.to_node) == 0:
            return self.prob.spec.get_output_flag(node.output_shape)
        
        return None 

    def _get_stability_flag(self, node: Node):
        stability = node.get_propagation_count()
        if stability / node.output_size >= self.config.SIP.STABILITY_FLAG_THRESHOLD:
            return  None
            
        return node.get_propagation_flag()

    def back_substitution(
        self,
        eq: Equation,
        node: Node,
        instability_flag: torch.Tensor=None,
        lower_slopes: dict=None,
        upper_slopes: dict=None,
        os_sip: OSSIP=None
    ):
        """
        Substitutes the variables in an equation with input variables.

        Arguments:
            eq:
                The equation.
            node:
                The input node to the node corresponding to the equation.
            instability_flag:
                Flag of unstable nodes from IA.
            slopes:
                Relu slopes to use. If none then the default of minimum area
                approximation are used.

        Returns:
            The concrete bounds of the equation after back_substitution of the
            variables.
        """
        eqs_lower, eqs_upper, instability_flag = self._back_substitution(
            eq, eq.copy(), node, node.from_node[0], instability_flag, lower_slopes, upper_slopes, os_sip
        )

        if eqs_lower is None:
            return torch.empty(0), torch.empty(0), instability_flag

        sum_eq_lower, sum_eq_upper = eqs_lower[0], eqs_upper[0]
        for i in range(1, len(eqs_lower)):
            sum_eq_lower = sum_eq_lower.add(eqs_lower[i])
            sum_eq_upper = sum_eq_upper.add(eqs_upper[i])

        if self.config.SIP.ONE_STEP_SYMBOLIC is True and instability_flag is not None:
            update = instability_flag.flatten()
            os_sip.current_lower_eq.matrix[update, :] = sum_eq_lower.matrix
            os_sip.current_lower_eq.const[update] = sum_eq_lower.const
            os_sip.current_upper_eq.matrix[update, :] = sum_eq_upper.matrix
            os_sip.current_upper_eq.const[update] = sum_eq_upper.const

        lower_bounds = sum_eq_lower.concrete_values(
            self.prob.spec.input_node.bounds.lower.flatten(),
            self.prob.spec.input_node.bounds.upper.flatten(),
            'lower'
        )

        upper_bounds = sum_eq_upper.concrete_values(
            self.prob.spec.input_node.bounds.lower.flatten(),
            self.prob.spec.input_node.bounds.upper.flatten(),
            'upper'
        )

        return lower_bounds, upper_bounds, instability_flag


    def _back_substitution(
        self,
        lower_eq: Equation,
        upper_eq: Equation,
        base_node: Node,
        cur_node: Node,
        instability_flag: torch.Tensor=None,
        lower_slopes: torch.Tensor=None,
        upper_slopes: torch.Tensor=None,
        os_sip: OSSIP=None
    ):
        """
        Helper function for back_substitution
        """
        if isinstance(cur_node, Input):
            return  [lower_eq], [upper_eq], instability_flag

        _tranposed = False

        if cur_node.has_relu_activation() or isinstance(cur_node, MaxPool):
            if lower_slopes is not None and cur_node.to_cur_node[0].id in lower_slopes:
                node_l_slopes = lower_slopes[cur_node.to_cur_node[0].id]
            else:
                node_l_slopes = None
            if upper_slopes is not None and cur_node.to_cur_node[0].id in upper_slopes:
                node_u_slopes = upper_slopes[cur_node.to_cur_node[0].id]
            else:
                node_u_slopes = None
                
            in_flag = self._get_stability_flag(cur_node.from_node[0])
            out_flag = self._get_stability_flag(cur_node)
            lower_eq = lower_eq.interval_transpose(
                cur_node, 'lower', out_flag, in_flag, node_l_slopes
            )
            upper_eq = upper_eq.interval_transpose(
                cur_node, 'upper', out_flag, in_flag, node_u_slopes
            )
            _tranposed = True

        elif type(cur_node) in [Relu, Flatten, Unsqueeze, Reshape]:
            lower_eq = [lower_eq]
            upper_eq = [upper_eq]

        else:
            in_flag = self._get_stability_flag(cur_node.from_node[0])
            out_flag = self._get_stability_flag(cur_node)
            lower_eq = lower_eq.transpose(cur_node, out_flag, in_flag)
            upper_eq = upper_eq.transpose(cur_node, out_flag, in_flag)
            _tranposed = True

        lower_eqs, upper_eqs = [], []

        for i in range(len(lower_eq)):
            if instability_flag is not None and _tranposed is True:
            # and cur_node.depth == 1 and bound == 'lower':
                lower_eq[i], upper_eq[i], instability_flag = self._update_back_subs_eqs(
                    lower_eq[i],
                    upper_eq[i],
                    base_node,
                    cur_node,
                    cur_node.from_node[i],
                    instability_flag,
                    os_sip
                )
                if torch.sum(instability_flag) == 0:
                    return None, None, instability_flag

            back_subs_l_eq, back_subs_u_eq, instability_flag = self._back_substitution(
                lower_eq[i],
                upper_eq[i],
                base_node,
                cur_node.from_node[0],
                instability_flag=instability_flag,
                lower_slopes=lower_slopes,
                upper_slopes=upper_slopes,
                os_sip=os_sip
            )
            if back_subs_l_eq is None:
                return None, None, instability_flag
        
            lower_eqs.extend(back_subs_l_eq)
            upper_eqs.extend(back_subs_u_eq)

        return lower_eqs, upper_eqs, instability_flag

    def _update_back_subs_eqs(
        self,
        lower_equation: Equation,
        upper_equation: Equation,
        base_node: Node,
        cur_node: Node,
        input_node: Node,
        instability_flag: torch.Tensor,
        os_sip: OSSIP
    ) -> Equation:
        if base_node.has_relu_activation() is not True:
            return lower_equation, upper_equation, instability_flag
        assert base_node.has_relu_activation() is True, \
            "Update concerns only nodes with relu activation."
       
        if input_node.id not in os_sip.lower_eq:
            return lower_equation, upper_equation, instability_flag 
            
        lower_bounds = lower_equation.interval_dot(
            'lower', os_sip.lower_eq[input_node.id], os_sip.upper_eq[input_node.id]
        ).min_values(
            self.prob.spec.input_node.bounds.lower.flatten(),
            self.prob.spec.input_node.bounds.upper.flatten()
        )
        upper_bounds = upper_equation.interval_dot(
            'upper', os_sip.lower_eq[input_node.id], os_sip.upper_eq[input_node.id]
        ).max_values(
            self.prob.spec.input_node.bounds.lower.flatten(),
            self.prob.spec.input_node.bounds.upper.flatten()
        )

        unstable_idxs = torch.logical_and(lower_bounds < 0, upper_bounds > 0)
        stable_idxs = torch.logical_not(unstable_idxs)
        flag = torch.zeros(base_node.output_shape, dtype=torch.bool)
        flag[instability_flag] = stable_idxs
        base_node.bounds.lower[flag] = lower_bounds[stable_idxs] 
        base_node.bounds.upper[flag] = upper_bounds[stable_idxs] 
        # print('-', base_node.to_node[0].get_unstable_count())
        base_node.to_node[0].reset_state_flags()
        # print('l', base_node.to_node[0].get_unstable_count())
        # input()
 
        flag = torch.zeros(base_node.output_shape, dtype=torch.bool)
        flag[instability_flag] = unstable_idxs
        reduced_lower_eq = Equation(
            lower_equation.matrix[unstable_idxs, :],
            lower_equation.const[unstable_idxs],
            self.config
        )   
        reduced_upper_eq = Equation(
            upper_equation.matrix[unstable_idxs, :],
            upper_equation.const[unstable_idxs],
            self.config
        )

        return reduced_lower_eq, reduced_upper_eq, flag

