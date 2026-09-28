"""Education primitives for courses, learners, mastery, and progress."""

from .mastery import MasteryTracker
from .grading import AssessmentEngine, AssessmentGrade, QuestionGrade, grade_question
from .creativity import CreativeChallenge, CreativityEngine
from .concept_graph import ConceptGraph
from .coding import CodingEvaluation, CodingEvaluator, CodingExercise
from .coding_feedback import CodeQualityReport, CodingFeedback, CodingFeedbackEngine
from .collaboration import CollaborationMessage, PairProgrammingSession
from .analytics import InstructorDashboard, LearnerAnalytics, LearningAnalytics, build_dashboard
from .anti_cheating import AntiCheatingAnalyzer, AntiCheatingReport
from .billing import CourseEntitlement, EntitlementManager
from .certificates import Certificate, CertificateIssuer
from .projects import ProjectAssignment, ProjectMilestone
from .code_analysis import CodeSimilarity, CodeSimilarityAnalyzer, SimilarSubmission
from .course_context import CourseContext
from .hints import Hint, HintLadder
from .difficulty import DifficultyEstimate, DifficultyEstimator
from .knowledge_tracing import BayesianKnowledgeTracer, KnowledgeState
from .models import (
    AssessmentResult,
    Course,
    LearnerProfile,
    Lesson,
    ProgressEvent,
)
from .progress import ProgressTracker
from .questions import Assessment, Question, QuestionType
from .service import EducationService
from .pedagogy import PedagogyEngine, TeachingMode, TeachingPlan, TeachingRequest
from .reflections import ReflectionEngine, ReflectionPrompt
from .recommendations import RecommendationEngine
from .tutoring import Tutor, TutorResponse
from .tutor_service import TutorAnswer, TutorService
from .storage import EducationStore
from .sandbox import CodeSandbox, DockerPythonSandbox, SandboxLimits, SandboxResult
from .strategy_learning import StrategySelection, TeachingStrategyBandit
from .spaced_repetition import ReviewSchedule, SpacedRepetitionScheduler

__all__ = [
    "AssessmentResult",
    "Assessment",
    "AssessmentEngine",
    "AssessmentGrade",
    "CreativeChallenge",
    "CreativityEngine",
    "CourseContext",
    "ConceptGraph",
    "CodeSandbox",
    "CodeQualityReport",
    "CodingFeedback",
    "CodingFeedbackEngine",
    "CodingEvaluation",
    "CodingEvaluator",
    "CodingExercise",
    "CollaborationMessage",
    "PairProgrammingSession",
    "LearningAnalytics",
    "LearnerAnalytics",
    "InstructorDashboard",
    "build_dashboard",
    "AntiCheatingAnalyzer",
    "AntiCheatingReport",
    "CourseEntitlement",
    "EntitlementManager",
    "Certificate",
    "CertificateIssuer",
    "ProjectAssignment",
    "ProjectMilestone",
    "DockerPythonSandbox",
    "CodeSimilarity",
    "CodeSimilarityAnalyzer",
    "DifficultyEstimate",
    "DifficultyEstimator",
    "Course",
    "EducationStore",
    "EducationService",
    "Hint",
    "HintLadder",
    "LearnerProfile",
    "BayesianKnowledgeTracer",
    "KnowledgeState",
    "Lesson",
    "MasteryTracker",
    "ProgressEvent",
    "ProgressTracker",
    "RecommendationEngine",
    "PedagogyEngine",
    "Question",
    "QuestionGrade",
    "QuestionType",
    "ReflectionEngine",
    "ReflectionPrompt",
    "ReviewSchedule",
    "SpacedRepetitionScheduler",
    "SandboxLimits",
    "SandboxResult",
    "SimilarSubmission",
    "StrategySelection",
    "TeachingStrategyBandit",
    "TeachingMode",
    "TeachingPlan",
    "TeachingRequest",
    "Tutor",
    "TutorAnswer",
    "TutorService",
    "TutorResponse",
    "grade_question",
]
