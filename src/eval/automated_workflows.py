"""
Automated Evaluation Workflows

This module provides automated workflow management for evaluation pipelines,
including scheduled evaluations, continuous monitoring, and automated reporting.

Key Features:
- Scheduled evaluation workflows
- Continuous performance monitoring  
- Automated alerting and notifications
- Workflow orchestration and dependency management
- Integration with existing evaluation components
- Automated report generation and distribution
"""

import uuid
import json
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
from collections import deque
import concurrent.futures

from agents.utils import AgentLogger
from .comprehensive_evaluator import AutomatedEvaluationPipeline, EvaluationBatch, ABTestResult
from .statistical_analyzer import StatisticalAnalyzer, EvaluationReportGenerator


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TriggerType(Enum):
    """Types of workflow triggers."""
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    MANUAL = "manual"
    THRESHOLD = "threshold"
    DEPENDENCY = "dependency"


@dataclass
class WorkflowTrigger:
    """
    Configuration for workflow triggers.
    """
    trigger_type: TriggerType
    schedule: Optional[str] = None  # Cron-like schedule
    event_type: Optional[str] = None  # Event type to listen for
    threshold_metric: Optional[str] = None  # Metric to monitor
    threshold_value: Optional[float] = None  # Threshold value
    threshold_operator: Optional[str] = None  # "greater", "less", "equal"
    dependencies: List[str] = field(default_factory=list)  # Workflow IDs to depend on
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trigger_type': self.trigger_type.value,
            'schedule': self.schedule,
            'event_type': self.event_type,
            'threshold_metric': self.threshold_metric,
            'threshold_value': self.threshold_value,
            'threshold_operator': self.threshold_operator,
            'dependencies': self.dependencies
        }


@dataclass
class WorkflowStep:
    """
    Individual step in a workflow.
    """
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    step_type: str = "evaluation"  # evaluation, analysis, report, notification
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # Step IDs this depends on
    timeout: Optional[int] = None  # Timeout in seconds
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_id': self.step_id,
            'name': self.name,
            'step_type': self.step_type,
            'config': self.config,
            'depends_on': self.depends_on,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


@dataclass
class WorkflowDefinition:
    """
    Complete workflow definition.
    """
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    trigger: WorkflowTrigger = field(default_factory=lambda: WorkflowTrigger(TriggerType.MANUAL))
    steps: List[WorkflowStep] = field(default_factory=list)
    notification_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'trigger': self.trigger.to_dict(),
            'steps': [step.to_dict() for step in self.steps],
            'notification_config': self.notification_config,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class WorkflowExecution:
    """
    Runtime execution state of a workflow.
    """
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'execution_id': self.execution_id,
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'step_results': self.step_results,
            'error_message': self.error_message,
            'progress': self.progress
        }


class WorkflowScheduler:
    """
    Manages scheduled execution of workflows.
    """
    
    def __init__(self):
        """Initialize workflow scheduler."""
        self.logger = AgentLogger("workflow_scheduler")
        self._schedules: Dict[str, WorkflowDefinition] = {}
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
    def add_scheduled_workflow(self, workflow: WorkflowDefinition) -> None:
        """
        Add a workflow to the scheduler.
        
        Args:
            workflow: WorkflowDefinition with scheduling information
        """
        if workflow.trigger.trigger_type == TriggerType.SCHEDULED:
            self._schedules[workflow.workflow_id] = workflow
            self.logger.log_debug(f"Added scheduled workflow '{workflow.name}'")
        else:
            raise ValueError("Workflow must have SCHEDULED trigger type")
    
    def remove_scheduled_workflow(self, workflow_id: str) -> bool:
        """
        Remove a workflow from the scheduler.
        
        Args:
            workflow_id: ID of workflow to remove
            
        Returns:
            True if workflow was removed, False if not found
        """
        return self._schedules.pop(workflow_id, None) is not None
    
    def start_scheduler(self) -> None:
        """Start the scheduler thread."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        self.logger.log_debug("Workflow scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler thread."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        self.logger.log_debug("Workflow scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                current_time = datetime.now()
                
                for workflow_id, workflow in list(self._schedules.items()):
                    if self._should_execute_workflow(workflow, current_time):
                        # Trigger workflow execution (this would integrate with WorkflowOrchestrator)
                        self.logger.log_debug(f"Triggering scheduled workflow: {workflow.name}")
                        # TODO: Integrate with workflow executor
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.log_error(e, "Scheduler loop error")
                time.sleep(60)
    
    def _should_execute_workflow(self, workflow: WorkflowDefinition, current_time: datetime) -> bool:
        """
        Check if workflow should be executed based on its schedule.
        
        Args:
            workflow: Workflow to check
            current_time: Current timestamp
            
        Returns:
            True if workflow should be executed
        """
        # This is a simplified implementation
        # In production, would use cron-like scheduling library
        if not workflow.trigger.schedule:
            return False
        
        # Parse simple schedule formats
        schedule = workflow.trigger.schedule.lower()
        
        if schedule == "hourly":
            return current_time.minute == 0
        elif schedule == "daily":
            return current_time.hour == 0 and current_time.minute == 0
        elif schedule == "weekly":
            return current_time.weekday() == 0 and current_time.hour == 0 and current_time.minute == 0
        
        return False


class WorkflowOrchestrator:
    """
    Orchestrates execution of evaluation workflows.
    """
    
    def __init__(self, 
                 evaluation_pipeline: Optional[AutomatedEvaluationPipeline] = None,
                 statistical_analyzer: Optional[StatisticalAnalyzer] = None,
                 report_generator: Optional[EvaluationReportGenerator] = None):
        """
        Initialize workflow orchestrator.
        
        Args:
            evaluation_pipeline: Evaluation pipeline instance
            statistical_analyzer: Statistical analyzer instance
            report_generator: Report generator instance
        """
        self.evaluation_pipeline = evaluation_pipeline or AutomatedEvaluationPipeline()
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer()
        self.report_generator = report_generator or EvaluationReportGenerator()
        
        self.logger = AgentLogger("workflow_orchestrator")
        
        # Workflow management
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: deque = deque(maxlen=1000)
        
        # Event system
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Scheduler
        self.scheduler = WorkflowScheduler()
        
    def register_workflow(self, workflow: WorkflowDefinition) -> str:
        """
        Register a workflow definition.
        
        Args:
            workflow: WorkflowDefinition to register
            
        Returns:
            Workflow ID
        """
        self.workflow_definitions[workflow.workflow_id] = workflow
        
        # Add to scheduler if it's scheduled
        if workflow.trigger.trigger_type == TriggerType.SCHEDULED:
            self.scheduler.add_scheduled_workflow(workflow)
        
        self.logger.log_debug(f"Registered workflow '{workflow.name}' ({workflow.workflow_id})")
        return workflow.workflow_id
    
    def execute_workflow(self, workflow_id: str, manual_trigger: bool = False) -> WorkflowExecution:
        """
        Execute a workflow.
        
        Args:
            workflow_id: ID of workflow to execute
            manual_trigger: Whether this is a manual execution
            
        Returns:
            WorkflowExecution instance
        """
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflow_definitions[workflow_id]
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            started_at=datetime.now()
        )
        
        self.active_executions[execution.execution_id] = execution
        
        try:
            self._execute_workflow_steps(workflow, execution)
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.progress = 1.0
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            self.logger.log_error(e, "Workflow execution failed")
        
        finally:
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution.execution_id]
        
        return execution
    
    def _execute_workflow_steps(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> None:
        """Execute workflow steps in dependency order."""
        execution.status = WorkflowStatus.RUNNING
        
        # Build dependency graph
        steps_by_id = {step.step_id: step for step in workflow.steps}
        completed_steps = set()
        
        total_steps = len(workflow.steps)
        
        while len(completed_steps) < total_steps:
            # Find steps ready to execute
            ready_steps = []
            for step in workflow.steps:
                if (step.step_id not in completed_steps and
                    all(dep_id in completed_steps for dep_id in step.depends_on)):
                    ready_steps.append(step)
            
            if not ready_steps:
                raise RuntimeError("Workflow has circular dependencies or orphaned steps")
            
            # Execute ready steps
            for step in ready_steps:
                self._execute_workflow_step(step, execution)
                completed_steps.add(step.step_id)
                execution.progress = len(completed_steps) / total_steps
    
    def _execute_workflow_step(self, step: WorkflowStep, execution: WorkflowExecution) -> None:
        """Execute a single workflow step."""
        self.logger.log_debug(f"Executing step: {step.name}")
        
        try:
            if step.step_type == "evaluation":
                result = self._execute_evaluation_step(step)
            elif step.step_type == "analysis":
                result = self._execute_analysis_step(step)
            elif step.step_type == "report":
                result = self._execute_report_step(step)
            elif step.step_type == "notification":
                result = self._execute_notification_step(step)
            else:
                raise ValueError(f"Unknown step type: {step.step_type}")
            
            execution.step_results[step.step_id] = {
                'status': 'completed',
                'result': result,
                'completed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            step.retry_count += 1
            
            if step.retry_count <= step.max_retries:
                self.logger.log_debug(f"Step {step.name} failed, retrying ({step.retry_count}/{step.max_retries})")
                time.sleep(2 ** step.retry_count)  # Exponential backoff
                self._execute_workflow_step(step, execution)  # Retry
            else:
                execution.step_results[step.step_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }
                raise RuntimeError(f"Step {step.name} failed after {step.max_retries} retries: {e}")
    
    def _execute_evaluation_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute an evaluation step."""
        config = step.config
        
        # Create evaluation batch
        batch = self.evaluation_pipeline.create_evaluation_batch(
            name=config.get('name', f'Workflow_{step.step_id}'),
            questions=config.get('questions', []),
            configs=config.get('evaluation_configs', []),
            description=config.get('description', '')
        )
        
        # Run evaluation
        results = self.evaluation_pipeline.run_evaluation_batch(batch.batch_id)
        
        return {
            'batch_id': batch.batch_id,
            'results_summary': results.get('summary_statistics', {}),
            'evaluation_count': len(results.get('evaluation_results', []))
        }
    
    def _execute_analysis_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a statistical analysis step."""
        config = step.config
        
        # Get data from previous steps or external source
        data_source = config.get('data_source', 'previous_step')
        
        if data_source == 'previous_step':
            # Find previous evaluation step
            prev_step_id = step.depends_on[0] if step.depends_on else None
            if not prev_step_id:
                raise ValueError("Analysis step needs dependency on evaluation step")
            # In real implementation, would get data from execution context
            data = []  # Placeholder
        else:
            # Load from external source
            data = []  # Placeholder
        
        # Perform analysis
        analysis_type = config.get('analysis_type', 'summary')
        
        if analysis_type == 'summary':
            # Summary statistics for each metric
            results = {}
            # Implementation would analyze actual data
        elif analysis_type == 'trend':
            # Trend analysis
            results = {}
        elif analysis_type == 'correlation':
            # Correlation analysis
            results = {}
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        return {
            'analysis_type': analysis_type,
            'results': results,
            'data_points_analyzed': len(data)
        }
    
    def _execute_report_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a report generation step."""
        config = step.config
        
        # Get data from previous steps
        data = []  # Placeholder - would gather from execution context
        
        # Generate report
        report_content = self.report_generator.generate_comprehensive_report(data, config)
        
        # Save report
        output_path = config.get('output_path', 'reports/')
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"evaluation_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return {
            'report_file': str(report_file),
            'report_length': len(report_content),
            'generated_at': datetime.now().isoformat()
        }
    
    def _execute_notification_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a notification step."""
        config = step.config
        
        # Prepare notification content
        notification_type = config.get('type', 'log')
        message = config.get('message', 'Workflow step completed')
        
        if notification_type == 'log':
            self.logger.log_debug(f"Notification: {message}")
        elif notification_type == 'email':
            # Would integrate with email service
            self.logger.log_debug(f"Email notification: {message}")
        elif notification_type == 'webhook':
            # Would send webhook
            self.logger.log_debug(f"Webhook notification: {message}")
        
        return {
            'notification_type': notification_type,
            'message': message,
            'sent_at': datetime.now().isoformat()
        }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow and its recent executions."""
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflow_definitions[workflow_id]
        
        # Find recent executions
        recent_executions = [
            exec.to_dict() for exec in self.execution_history
            if exec.workflow_id == workflow_id
        ][-10:]  # Last 10 executions
        
        return {
            'workflow': workflow.to_dict(),
            'recent_executions': recent_executions,
            'active_executions': [
                exec.to_dict() for exec in self.active_executions.values()
                if exec.workflow_id == workflow_id
            ]
        }
    
    def start(self) -> None:
        """Start the workflow orchestrator."""
        self.scheduler.start_scheduler()
        self.logger.log_debug("Workflow orchestrator started")
    
    def stop(self) -> None:
        """Stop the workflow orchestrator."""
        self.scheduler.stop_scheduler()
        self.logger.log_debug("Workflow orchestrator stopped")


class ContinuousMonitoringService:
    """
    Service for continuous monitoring of evaluation metrics.
    """
    
    def __init__(self, orchestrator: WorkflowOrchestrator):
        """
        Initialize monitoring service.
        
        Args:
            orchestrator: WorkflowOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.logger = AgentLogger("monitoring_service")
        
        # Monitoring configuration
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.check_interval = 300  # 5 minutes
        
        # Metric thresholds
        self.thresholds: Dict[str, Dict[str, Any]] = {}
        
        # Alert history
        self.alert_history: deque = deque(maxlen=1000)
    
    def add_metric_threshold(self, 
                           metric_name: str, 
                           threshold_value: float,
                           operator: str = "less",
                           alert_workflow_id: Optional[str] = None) -> None:
        """
        Add a metric threshold for monitoring.
        
        Args:
            metric_name: Name of metric to monitor
            threshold_value: Threshold value
            operator: Comparison operator ("less", "greater", "equal")
            alert_workflow_id: Workflow to trigger when threshold is breached
        """
        self.thresholds[metric_name] = {
            'threshold_value': threshold_value,
            'operator': operator,
            'alert_workflow_id': alert_workflow_id,
            'last_checked': None,
            'breach_count': 0
        }
        
        self.logger.log_debug(f"Added threshold for {metric_name}: {operator} {threshold_value}")
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.log_debug("Continuous monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.log_debug("Continuous monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                current_time = datetime.now()
                
                for metric_name, threshold_config in self.thresholds.items():
                    self._check_metric_threshold(metric_name, threshold_config, current_time)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.log_error(e, "Monitoring loop error")
                time.sleep(60)
    
    def _check_metric_threshold(self, 
                              metric_name: str, 
                              threshold_config: Dict[str, Any], 
                              current_time: datetime) -> None:
        """Check if a metric threshold has been breached."""
        try:
            # Get recent metric value (this would integrate with actual data source)
            current_value = self._get_current_metric_value(metric_name)
            
            if current_value is None:
                return
            
            # Check threshold
            threshold_value = threshold_config['threshold_value']
            operator = threshold_config['operator']
            
            threshold_breached = False
            if operator == "less" and current_value < threshold_value:
                threshold_breached = True
            elif operator == "greater" and current_value > threshold_value:
                threshold_breached = True
            elif operator == "equal" and abs(current_value - threshold_value) < 0.001:
                threshold_breached = True
            
            if threshold_breached:
                threshold_config['breach_count'] += 1
                
                # Create alert
                alert = {
                    'metric_name': metric_name,
                    'current_value': current_value,
                    'threshold_value': threshold_value,
                    'operator': operator,
                    'breach_count': threshold_config['breach_count'],
                    'timestamp': current_time.isoformat()
                }
                
                self.alert_history.append(alert)
                self.logger.log_debug(f"Threshold breach: {metric_name} = {current_value}")
                
                # Trigger alert workflow if configured
                alert_workflow_id = threshold_config.get('alert_workflow_id')
                if alert_workflow_id:
                    self.orchestrator.execute_workflow(alert_workflow_id, manual_trigger=True)
            
            threshold_config['last_checked'] = current_time
            
        except Exception as e:
            self.logger.log_error(e, f"Error checking threshold for {metric_name}")
    
    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric."""
        # This would integrate with actual data sources
        # For now, return a placeholder value
        return 0.75  # Placeholder
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'check_interval': self.check_interval,
            'thresholds_configured': len(self.thresholds),
            'recent_alerts': list(self.alert_history)[-10:],  # Last 10 alerts
            'threshold_details': self.thresholds
        }


# Factory functions for workflow creation

def create_basic_evaluation_workflow(name: str, 
                                   questions: List[str], 
                                   schedule: str = "daily") -> WorkflowDefinition:
    """
    Create a basic evaluation workflow.
    
    Args:
        name: Workflow name
        questions: Questions to evaluate
        schedule: Execution schedule
        
    Returns:
        WorkflowDefinition
    """
    # Evaluation step
    eval_step = WorkflowStep(
        name="Run Evaluation",
        step_type="evaluation",
        config={
            'name': f'{name}_evaluation',
            'questions': questions,
            'evaluation_configs': [
                {'type': 'quality', 'id': 'quality_eval'},
                {'type': 'dialectical', 'id': 'dialectical_eval'}
            ]
        }
    )
    
    # Analysis step
    analysis_step = WorkflowStep(
        name="Analyze Results",
        step_type="analysis",
        config={'analysis_type': 'summary'},
        depends_on=[eval_step.step_id]
    )
    
    # Report step
    report_step = WorkflowStep(
        name="Generate Report",
        step_type="report",
        config={'output_path': 'reports/'},
        depends_on=[analysis_step.step_id]
    )
    
    return WorkflowDefinition(
        name=name,
        description=f"Basic evaluation workflow for {len(questions)} questions",
        trigger=WorkflowTrigger(TriggerType.SCHEDULED, schedule=schedule),
        steps=[eval_step, analysis_step, report_step]
    )


def create_monitoring_workflow(metric_name: str, 
                             threshold: float, 
                             operator: str = "less") -> WorkflowDefinition:
    """
    Create a monitoring workflow triggered by metric thresholds.
    
    Args:
        metric_name: Metric to monitor
        threshold: Threshold value
        operator: Comparison operator
        
    Returns:
        WorkflowDefinition
    """
    # Analysis step
    analysis_step = WorkflowStep(
        name="Analyze Metric Breach",
        step_type="analysis",
        config={
            'analysis_type': 'trend',
            'metric_name': metric_name
        }
    )
    
    # Notification step
    notification_step = WorkflowStep(
        name="Send Alert",
        step_type="notification",
        config={
            'type': 'log',
            'message': f'Metric {metric_name} breached threshold {threshold}'
        },
        depends_on=[analysis_step.step_id]
    )
    
    return WorkflowDefinition(
        name=f"Monitor {metric_name}",
        description=f"Monitor {metric_name} for threshold breaches",
        trigger=WorkflowTrigger(
            TriggerType.THRESHOLD,
            threshold_metric=metric_name,
            threshold_value=threshold,
            threshold_operator=operator
        ),
        steps=[analysis_step, notification_step]
    )


# Export main classes
__all__ = [
    'WorkflowStatus',
    'TriggerType', 
    'WorkflowTrigger',
    'WorkflowStep',
    'WorkflowDefinition',
    'WorkflowExecution',
    'WorkflowScheduler',
    'WorkflowOrchestrator',
    'ContinuousMonitoringService',
    'create_basic_evaluation_workflow',
    'create_monitoring_workflow'
]