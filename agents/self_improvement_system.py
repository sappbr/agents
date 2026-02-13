"""
Self-Improvement System for Trading Strategy Optimization
Implements AGI Gate 8: Recursive Self-Improvement

This module provides automated strategy optimization, parameter tuning,
and architecture improvements based on backtest performance.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error
import optuna
from agents.agi_evolution_framework import get_agi_framework, GateLevel

@dataclass
class OptimizationResult:
    """Result of an optimization run"""
    parameters: Dict[str, Any]
    performance_score: float
    improvement_pct: float
    backtest_results: Dict[str, Any]
    timestamp: datetime
    gate_improvements: Dict[int, float]

class SelfImprovementSystem:
    """
    Automated self-improvement system for trading strategies.
    Implements AGI Gate 8 capabilities for recursive optimization.
    """

    def __init__(self, workspace_path: str = "/workspaces/agents"):
        self.workspace_path = workspace_path
        self.agi_framework = get_agi_framework()

        # Optimization history
        self.optimization_history_file = os.path.join(workspace_path, "optimization_history.json")
        self.optimization_history = self._load_optimization_history()

        # Current best parameters
        self.best_parameters = self._get_default_parameters()
        self.best_score = 0.0

        # Optimization bounds and constraints
        self.parameter_bounds = {
            'max_position_size': (0.05, 0.25),      # 5% to 25% of capital
            'max_total_risk': (0.3, 0.7),           # 30% to 70% total exposure
            'stop_loss_pct': (0.02, 0.10),          # 2% to 10% stop loss
            'take_profit_pct': (0.10, 0.40),        # 10% to 40% take profit
            'confidence_threshold': (0.6, 0.9),     # AI confidence threshold
            'max_daily_trades': (5, 20),            # Daily trade limits
            'cool_down_period': (180, 600),         # 3-10 minutes between trades
        }

    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default trading parameters"""
        return {
            'max_position_size': 0.10,
            'max_total_risk': 0.50,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.20,
            'confidence_threshold': 0.70,
            'max_daily_trades': 10,
            'cool_down_period': 300,
        }

    def _load_optimization_history(self) -> List[OptimizationResult]:
        """Load optimization history from file"""
        if os.path.exists(self.optimization_history_file):
            with open(self.optimization_history_file, 'r') as f:
                data = json.load(f)
                history = []
                for item in data:
                    history.append(OptimizationResult(
                        parameters=item['parameters'],
                        performance_score=item['performance_score'],
                        improvement_pct=item['improvement_pct'],
                        backtest_results=item['backtest_results'],
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        gate_improvements=item.get('gate_improvements', {})
                    ))
                return history
        return []

    def _save_optimization_history(self):
        """Save optimization history to file"""
        data = []
        for result in self.optimization_history:
            data.append({
                'parameters': result.parameters,
                'performance_score': result.performance_score,
                'improvement_pct': result.improvement_pct,
                'backtest_results': result.backtest_results,
                'timestamp': result.timestamp.isoformat(),
                'gate_improvements': result.gate_improvements
            })

        with open(self.optimization_history_file, 'w') as f:
            json.dump(data, f, indent=2)

    def run_automated_optimization(self, generations: int = 5, population_size: int = 10) -> OptimizationResult:
        """
        Run automated optimization using evolutionary algorithms and Bayesian optimization
        """
        print("🔄 Starting automated strategy optimization...")
        print(f"Generations: {generations}, Population size: {population_size}")

        best_result = None
        best_score = self.best_score

        for generation in range(generations):
            print(f"\n📊 Generation {generation + 1}/{generations}")

            # Generate parameter candidates
            candidates = self._generate_parameter_candidates(population_size)

            # Evaluate each candidate
            generation_results = []
            for i, params in enumerate(candidates):
                print(f"  Evaluating candidate {i+1}/{population_size}...")
                try:
                    score, backtest_results = self._evaluate_parameters(params)
                    generation_results.append((params, score, backtest_results))

                    if score > best_score:
                        best_score = score
                        improvement_pct = ((score - self.best_score) / self.best_score * 100) if self.best_score > 0 else 0

                        best_result = OptimizationResult(
                            parameters=params,
                            performance_score=score,
                            improvement_pct=improvement_pct,
                            backtest_results=backtest_results,
                            timestamp=datetime.now(),
                            gate_improvements=self._calculate_gate_improvements(params, backtest_results)
                        )

                        print(f"    🏆 New best score: {score:.4f} (+{improvement_pct:.1f}%)")

                except Exception as e:
                    print(f"    ❌ Evaluation failed: {e}")
                    continue

            # Update best parameters
            if best_result:
                self.best_parameters = best_result.parameters
                self.best_score = best_result.performance_score

                # Save to history
                self.optimization_history.append(best_result)
                self._save_optimization_history()

                # Update AGI framework
                self._update_agi_framework_with_improvements(best_result)

        if best_result:
            print("
✅ Optimization completed!"            print(f"Best parameters: {best_result.parameters}")
            print(".4f"            print(".1f"        else:
            print("\n❌ Optimization failed to find improvements")

        return best_result

    def _generate_parameter_candidates(self, population_size: int) -> List[Dict[str, Any]]:
        """Generate parameter candidates using various sampling methods"""
        candidates = []

        # Method 1: Random sampling around current best
        for _ in range(population_size // 2):
            candidate = {}
            for param, (min_val, max_val) in self.parameter_bounds.items():
                current_val = self.best_parameters.get(param, (min_val + max_val) / 2)
                # Add noise around current best
                noise = np.random.normal(0, 0.1)  # 10% standard deviation
                new_val = current_val * (1 + noise)
                new_val = max(min_val, min(max_val, new_val))
                candidate[param] = round(new_val, 4) if isinstance(new_val, float) else int(new_val)
            candidates.append(candidate)

        # Method 2: Grid search for key parameters
        key_params = ['max_position_size', 'stop_loss_pct', 'take_profit_pct']
        grid_candidates = self._generate_grid_candidates(key_params, 3)  # 3x3x3 = 27 combinations, but limit to population_size//2
        candidates.extend(grid_candidates[:population_size // 2])

        return candidates[:population_size]

    def _generate_grid_candidates(self, params: List[str], points_per_param: int) -> List[Dict[str, Any]]:
        """Generate grid search candidates for specified parameters"""
        candidates = []

        # Create grid points for each parameter
        param_grids = {}
        for param in params:
            min_val, max_val = self.parameter_bounds[param]
            param_grids[param] = np.linspace(min_val, max_val, points_per_param)

        # Generate all combinations
        import itertools
        for combination in itertools.product(*param_grids.values()):
            candidate = dict(zip(params, combination))
            # Fill in other parameters with current best
            for param in self.parameter_bounds.keys():
                if param not in candidate:
                    candidate[param] = self.best_parameters.get(param, self._get_default_parameters()[param])
            candidates.append(candidate)

        return candidates

    def _evaluate_parameters(self, parameters: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a set of parameters by running a backtest
        """
        from backtest_engine import BacktestEngine

        # Create engine with new parameters
        engine = BacktestEngine(start_capital=100.0, realistic_mode=True)

        # Apply custom parameters
        for param, value in parameters.items():
            if hasattr(engine, param):
                setattr(engine, param, value)

        # Run quick backtest
        historical_data = engine.download_historical_data(days=30)

        if not historical_data:
            raise ValueError("Failed to generate historical data")

        engine.run_backtest(historical_data, days=30, use_live_data=False)

        # Calculate composite score
        score = self._calculate_performance_score(engine)

        backtest_results = {
            'final_capital': engine.current_capital,
            'total_return': (engine.current_capital - 100.0) / 100.0,
            'total_trades': engine.total_trades,
            'win_rate': engine.winning_trades / engine.total_trades if engine.total_trades > 0 else 0,
            'max_drawdown': engine.max_drawdown,
            'sharpe_ratio': engine.sharpe_ratio,
            'total_pnl': engine.total_pnl
        }

        return score, backtest_results

    def _calculate_performance_score(self, engine) -> float:
        """
        Calculate a composite performance score for parameter evaluation
        """
        # Weighted combination of metrics
        weights = {
            'total_return': 0.3,
            'sharpe_ratio': 0.25,
            'win_rate': 0.2,
            'max_drawdown_penalty': 0.15,  # Negative weight
            'trade_frequency': 0.1
        }

        total_return = (engine.current_capital - 100.0) / 100.0
        sharpe = max(0, engine.sharpe_ratio)  # Only positive Sharpe contributes
        win_rate = engine.winning_trades / engine.total_trades if engine.total_trades > 0 else 0
        drawdown_penalty = min(0, -engine.max_drawdown * 2)  # Penalty for drawdowns
        trade_frequency = min(1.0, engine.total_trades / 20.0)  # Reward up to 20 trades

        score = (
            weights['total_return'] * total_return +
            weights['sharpe_ratio'] * sharpe +
            weights['win_rate'] * win_rate +
            weights['max_drawdown_penalty'] * drawdown_penalty +
            weights['trade_frequency'] * trade_frequency
        )

        # Normalize to 0-1 range (roughly)
        return max(0, min(1, score))

    def _calculate_gate_improvements(self, parameters: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[int, float]:
        """Calculate improvements in AGI gate metrics from parameter changes"""
        improvements = {}

        # Gate 3 (Causal Grounding): Better parameters = better causal understanding
        if backtest_results['sharpe_ratio'] > 1.0:
            improvements[3] = 0.1  # Small improvement in causal reasoning

        # Gate 6 (Emotional Architecture): Adaptive parameters show regime awareness
        if parameters.get('max_position_size', 0.1) != 0.1:  # Non-default position sizing
            improvements[6] = 0.05

        # Gate 8 (Self-Improvement): This optimization itself is self-improvement
        improvements[8] = 0.15  # Direct improvement from optimization

        return improvements

    def _update_agi_framework_with_improvements(self, result: OptimizationResult):
        """Update AGI framework with optimization results"""
        # Update Gate 8 (Self-Improvement)
        current_level = self.agi_framework.gates[8].current_level
        new_metrics = self.agi_framework.gates[8].metrics.copy()
        new_metrics['self_optimization_cycles'] = new_metrics.get('self_optimization_cycles', 0) + 1
        new_metrics['improvement_rate'] = result.improvement_pct / 100.0

        self.agi_framework.update_gate_assessment(
            8, current_level, new_metrics,
            f"Optimization cycle completed with {result.improvement_pct:.1f}% improvement"
        )

        # Update other gates based on improvements
        for gate_id, improvement in result.gate_improvements.items():
            if gate_id != 8:
                gate = self.agi_framework.gates[gate_id]
                self.agi_framework.update_gate_assessment(
                    gate_id, gate.current_level,
                    {'performance_improvement': improvement},
                    f"Parameter optimization improved gate {gate_id} performance by {improvement:.1%}"
                )

    def run_bayesian_optimization(self, n_trials: int = 50) -> OptimizationResult:
        """
        Run Bayesian optimization using Optuna for advanced parameter tuning
        """
        print("🎯 Starting Bayesian optimization...")

        def objective(trial):
            # Define parameter search space
            params = {
                'max_position_size': trial.suggest_float('max_position_size', 0.05, 0.25),
                'max_total_risk': trial.suggest_float('max_total_risk', 0.3, 0.7),
                'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.02, 0.10),
                'take_profit_pct': trial.suggest_float('take_profit_pct', 0.10, 0.40),
                'confidence_threshold': trial.suggest_float('confidence_threshold', 0.6, 0.9),
                'max_daily_trades': trial.suggest_int('max_daily_trades', 5, 20),
                'cool_down_period': trial.suggest_int('cool_down_period', 180, 600),
            }

            try:
                score, backtest_results = self._evaluate_parameters(params)
                return score
            except:
                return 0.0  # Return 0 for failed evaluations

        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Get best result
        best_params = study.best_params
        best_score, best_backtest = self._evaluate_parameters(best_params)

        improvement_pct = ((best_score - self.best_score) / self.best_score * 100) if self.best_score > 0 else 0

        result = OptimizationResult(
            parameters=best_params,
            performance_score=best_score,
            improvement_pct=improvement_pct,
            backtest_results=best_backtest,
            timestamp=datetime.now(),
            gate_improvements=self._calculate_gate_improvements(best_params, best_backtest)
        )

        # Update system
        self.best_parameters = best_params
        self.best_score = best_score
        self.optimization_history.append(result)
        self._save_optimization_history()
        self._update_agi_framework_with_improvements(result)

        print("✅ Bayesian optimization completed!"        print(f"Best score: {best_score:.4f}")
        print(f"Improvement: {improvement_pct:.1f}%")

        return result

    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for further optimization"""
        recommendations = []

        # Analyze recent optimization history
        if len(self.optimization_history) < 3:
            recommendations.append("Run more optimization cycles to establish baseline performance")
            return recommendations

        recent_results = self.optimization_history[-5:]

        # Check for parameter sensitivity
        param_sensitivity = self._analyze_parameter_sensitivity(recent_results)
        if param_sensitivity:
            recommendations.append(f"Most sensitive parameters: {', '.join(param_sensitivity[:3])}")

        # Check for diminishing returns
        improvements = [r.improvement_pct for r in recent_results]
        if len(improvements) >= 3 and all(i < 1.0 for i in improvements[-3:]):
            recommendations.append("Optimization showing diminishing returns - consider architecture changes")

        # Gate-specific recommendations
        gate_8_level = self.agi_framework.gates[8].current_level.value
        if gate_8_level < 3:
            recommendations.append("Upgrade to Level 3 self-improvement for automated architecture search")

        return recommendations

    def _analyze_parameter_sensitivity(self, results: List[OptimizationResult]) -> List[str]:
        """Analyze which parameters have the most impact on performance"""
        if len(results) < 2:
            return []

        # Simple sensitivity analysis
        param_impacts = {}

        for param in self.parameter_bounds.keys():
            values = [r.parameters.get(param, 0) for r in results]
            scores = [r.performance_score for r in results]

            if len(set(values)) > 1:  # Only if parameter varies
                correlation = abs(np.corrcoef(values, scores)[0, 1])
                param_impacts[param] = correlation

        # Sort by impact
        sorted_params = sorted(param_impacts.items(), key=lambda x: x[1], reverse=True)
        return [param for param, _ in sorted_params]

# Global self-improvement system instance
self_improvement_system = SelfImprovementSystem()

def get_self_improvement_system() -> SelfImprovementSystem:
    """Get the global self-improvement system instance"""
    return self_improvement_system

if __name__ == "__main__":
    # Example usage
    system = get_self_improvement_system()

    print("🚀 Self-Improvement System Demo")
    print("=" * 40)

    # Run automated optimization
    result = system.run_automated_optimization(generations=2, population_size=4)

    if result:
        print("
📊 Optimization Results:"        print(f"Best Score: {result.performance_score:.4f}")
        print(f"Improvement: {result.improvement_pct:.1f}%")
        print(f"Parameters: {result.parameters}")

    # Get recommendations
    recommendations = system.get_optimization_recommendations()
    print("
💡 Recommendations:"    for rec in recommendations:
        print(f"- {rec}")