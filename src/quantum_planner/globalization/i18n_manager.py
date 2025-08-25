"""International (I18n) Manager for Quantum Task Planner.

Provides comprehensive internationalization support including:
- Multi-language support (en, es, fr, de, ja, zh)
- Locale-aware formatting
- Cultural adaptation
- Right-to-left language support
- Dynamic language switching
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import locale
import gettext


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


@dataclass
class LocaleInfo:
    """Information about a specific locale."""
    
    language_code: str
    language_name: str
    country_code: Optional[str] = None
    is_rtl: bool = False
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S" 
    number_format: str = "1,234.56"
    currency_symbol: str = "$"
    decimal_separator: str = "."
    thousands_separator: str = ","


class InternationalizationManager:
    """Main internationalization manager."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        """Initialize internationalization manager.
        
        Args:
            default_language: Default language to use
        """
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.locale_info: Dict[str, LocaleInfo] = {}
        self.message_catalogs: Dict[str, gettext.GNUTranslations] = {}
        
        # Initialize supported locales
        self._initialize_locales()
        
        # Load translations
        self._load_translations()
    
    def _initialize_locales(self):
        """Initialize supported locale information."""
        self.locale_info = {
            "en": LocaleInfo(
                language_code="en",
                language_name="English",
                country_code="US",
                date_format="%Y-%m-%d",
                time_format="%H:%M:%S",
                currency_symbol="$"
            ),
            "es": LocaleInfo(
                language_code="es", 
                language_name="Español",
                country_code="ES",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                currency_symbol="€"
            ),
            "fr": LocaleInfo(
                language_code="fr",
                language_name="Français", 
                country_code="FR",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                currency_symbol="€",
                decimal_separator=",",
                thousands_separator=" "
            ),
            "de": LocaleInfo(
                language_code="de",
                language_name="Deutsch",
                country_code="DE", 
                date_format="%d.%m.%Y",
                time_format="%H:%M",
                currency_symbol="€",
                decimal_separator=",",
                thousands_separator="."
            ),
            "ja": LocaleInfo(
                language_code="ja",
                language_name="日本語",
                country_code="JP",
                date_format="%Y年%m月%d日",
                time_format="%H時%M分",
                currency_symbol="¥"
            ),
            "zh": LocaleInfo(
                language_code="zh",
                language_name="中文",
                country_code="CN",
                date_format="%Y年%m月%d日", 
                time_format="%H:%M",
                currency_symbol="¥"
            )
        }
    
    def _load_translations(self):
        """Load translation files for all supported languages."""
        # Default translations for quantum task planner
        self.translations = {
            "en": {
                # Core messages
                "quantum_planner_title": "Quantum Task Planner",
                "task_assignment_complete": "Task assignment completed successfully",
                "optimization_in_progress": "Optimization in progress...",
                "solution_found": "Solution found",
                "no_solution_found": "No feasible solution found",
                
                # Agent messages
                "agent_created": "Agent '{name}' created with {skill_count} skills",
                "agent_assigned": "Agent '{agent}' assigned to task '{task}'",
                "agent_capacity_exceeded": "Agent capacity exceeded",
                "agent_unavailable": "Agent is currently unavailable",
                
                # Task messages  
                "task_created": "Task '{name}' created (Priority: {priority})",
                "task_completed": "Task '{name}' completed successfully",
                "task_failed": "Task '{name}' failed to complete",
                "task_duration": "Estimated duration: {duration} hours",
                
                # Optimization messages
                "quantum_backend_selected": "Quantum backend selected: {backend}",
                "classical_fallback": "Using classical fallback optimization",
                "optimization_converged": "Optimization converged after {iterations} iterations",
                "makespan_optimized": "Makespan optimized to {makespan} hours",
                
                # Error messages
                "invalid_input": "Invalid input provided",
                "insufficient_resources": "Insufficient resources for task assignment",
                "optimization_failed": "Optimization failed to converge",
                "backend_unavailable": "Quantum backend is unavailable",
                
                # Status messages
                "ready": "Ready",
                "running": "Running", 
                "completed": "Completed",
                "failed": "Failed",
                "cancelled": "Cancelled"
            },
            
            "es": {
                # Core messages
                "quantum_planner_title": "Planificador de Tareas Cuánticas",
                "task_assignment_complete": "Asignación de tareas completada exitosamente",
                "optimization_in_progress": "Optimización en progreso...",
                "solution_found": "Solución encontrada",
                "no_solution_found": "No se encontró solución factible",
                
                # Agent messages
                "agent_created": "Agente '{name}' creado con {skill_count} habilidades",
                "agent_assigned": "Agente '{agent}' asignado a tarea '{task}'",
                "agent_capacity_exceeded": "Capacidad del agente excedida",
                "agent_unavailable": "El agente no está disponible actualmente",
                
                # Task messages
                "task_created": "Tarea '{name}' creada (Prioridad: {priority})",
                "task_completed": "Tarea '{name}' completada exitosamente",
                "task_failed": "Tarea '{name}' falló al completarse", 
                "task_duration": "Duración estimada: {duration} horas",
                
                # Optimization messages
                "quantum_backend_selected": "Backend cuántico seleccionado: {backend}",
                "classical_fallback": "Usando optimización clásica de respaldo",
                "optimization_converged": "Optimización convergió después de {iterations} iteraciones",
                "makespan_optimized": "Makespan optimizado a {makespan} horas",
                
                # Error messages
                "invalid_input": "Entrada inválida proporcionada",
                "insufficient_resources": "Recursos insuficientes para asignación de tareas",
                "optimization_failed": "La optimización falló al converger",
                "backend_unavailable": "El backend cuántico no está disponible",
                
                # Status messages
                "ready": "Listo",
                "running": "Ejecutándose",
                "completed": "Completado", 
                "failed": "Falló",
                "cancelled": "Cancelado"
            },
            
            "fr": {
                # Core messages
                "quantum_planner_title": "Planificateur de Tâches Quantiques",
                "task_assignment_complete": "Attribution des tâches terminée avec succès",
                "optimization_in_progress": "Optimisation en cours...",
                "solution_found": "Solution trouvée",
                "no_solution_found": "Aucune solution réalisable trouvée",
                
                # Agent messages
                "agent_created": "Agent '{name}' créé avec {skill_count} compétences",
                "agent_assigned": "Agent '{agent}' assigné à la tâche '{task}'",
                "agent_capacity_exceeded": "Capacité de l'agent dépassée",
                "agent_unavailable": "L'agent n'est pas disponible actuellement",
                
                # Task messages
                "task_created": "Tâche '{name}' créée (Priorité: {priority})",
                "task_completed": "Tâche '{name}' terminée avec succès",
                "task_failed": "Tâche '{name}' a échoué",
                "task_duration": "Durée estimée: {duration} heures",
                
                # Optimization messages
                "quantum_backend_selected": "Backend quantique sélectionné: {backend}",
                "classical_fallback": "Utilisation de l'optimisation classique de secours",
                "optimization_converged": "Optimisation convergée après {iterations} itérations",
                "makespan_optimized": "Makespan optimisé à {makespan} heures",
                
                # Error messages
                "invalid_input": "Entrée invalide fournie",
                "insufficient_resources": "Ressources insuffisantes pour l'attribution des tâches",
                "optimization_failed": "L'optimisation a échoué à converger",
                "backend_unavailable": "Le backend quantique n'est pas disponible",
                
                # Status messages
                "ready": "Prêt",
                "running": "En cours",
                "completed": "Terminé",
                "failed": "Échoué", 
                "cancelled": "Annulé"
            },
            
            "de": {
                # Core messages
                "quantum_planner_title": "Quanten-Aufgabenplaner",
                "task_assignment_complete": "Aufgabenzuweisung erfolgreich abgeschlossen",
                "optimization_in_progress": "Optimierung läuft...",
                "solution_found": "Lösung gefunden",
                "no_solution_found": "Keine durchführbare Lösung gefunden",
                
                # Agent messages
                "agent_created": "Agent '{name}' erstellt mit {skill_count} Fähigkeiten",
                "agent_assigned": "Agent '{agent}' der Aufgabe '{task}' zugewiesen",
                "agent_capacity_exceeded": "Agentenkapazität überschritten",
                "agent_unavailable": "Agent ist derzeit nicht verfügbar",
                
                # Task messages
                "task_created": "Aufgabe '{name}' erstellt (Priorität: {priority})",
                "task_completed": "Aufgabe '{name}' erfolgreich abgeschlossen",
                "task_failed": "Aufgabe '{name}' konnte nicht abgeschlossen werden",
                "task_duration": "Geschätzte Dauer: {duration} Stunden",
                
                # Optimization messages
                "quantum_backend_selected": "Quanten-Backend ausgewählt: {backend}",
                "classical_fallback": "Verwende klassische Fallback-Optimierung",
                "optimization_converged": "Optimierung konvergierte nach {iterations} Iterationen",
                "makespan_optimized": "Makespan auf {makespan} Stunden optimiert",
                
                # Error messages
                "invalid_input": "Ungültige Eingabe bereitgestellt",
                "insufficient_resources": "Unzureichende Ressourcen für Aufgabenzuweisung",
                "optimization_failed": "Optimierung konvergierte nicht",
                "backend_unavailable": "Quanten-Backend ist nicht verfügbar",
                
                # Status messages
                "ready": "Bereit",
                "running": "Läuft",
                "completed": "Abgeschlossen",
                "failed": "Fehlgeschlagen",
                "cancelled": "Abgebrochen"
            },
            
            "ja": {
                # Core messages
                "quantum_planner_title": "量子タスクプランナー",
                "task_assignment_complete": "タスクの割り当てが正常に完了しました",
                "optimization_in_progress": "最適化中...",
                "solution_found": "解が見つかりました",
                "no_solution_found": "実行可能な解が見つかりません",
                
                # Agent messages
                "agent_created": "エージェント '{name}' が {skill_count} のスキルで作成されました",
                "agent_assigned": "エージェント '{agent}' がタスク '{task}' に割り当てられました",
                "agent_capacity_exceeded": "エージェントの容量を超過しました",
                "agent_unavailable": "エージェントは現在利用できません",
                
                # Task messages
                "task_created": "タスク '{name}' が作成されました（優先度: {priority}）",
                "task_completed": "タスク '{name}' が正常に完了しました",
                "task_failed": "タスク '{name}' の完了に失敗しました",
                "task_duration": "推定期間: {duration} 時間",
                
                # Optimization messages
                "quantum_backend_selected": "量子バックエンドが選択されました: {backend}",
                "classical_fallback": "古典的フォールバック最適化を使用",
                "optimization_converged": "最適化が {iterations} 回の反復後に収束しました",
                "makespan_optimized": "メイクスパンが {makespan} 時間に最適化されました",
                
                # Error messages
                "invalid_input": "無効な入力が提供されました",
                "insufficient_resources": "タスクの割り当てに十分なリソースがありません",
                "optimization_failed": "最適化の収束に失敗しました",
                "backend_unavailable": "量子バックエンドが利用できません",
                
                # Status messages
                "ready": "準備完了",
                "running": "実行中",
                "completed": "完了",
                "failed": "失敗",
                "cancelled": "キャンセル"
            },
            
            "zh": {
                # Core messages
                "quantum_planner_title": "量子任务规划器",
                "task_assignment_complete": "任务分配成功完成",
                "optimization_in_progress": "优化进行中...",
                "solution_found": "找到解决方案",
                "no_solution_found": "未找到可行解决方案",
                
                # Agent messages
                "agent_created": "代理 '{name}' 已创建，具有 {skill_count} 项技能",
                "agent_assigned": "代理 '{agent}' 已分配给任务 '{task}'",
                "agent_capacity_exceeded": "代理容量已超出",
                "agent_unavailable": "代理目前不可用",
                
                # Task messages
                "task_created": "任务 '{name}' 已创建（优先级: {priority}）",
                "task_completed": "任务 '{name}' 成功完成",
                "task_failed": "任务 '{name}' 完成失败",
                "task_duration": "预计持续时间: {duration} 小时",
                
                # Optimization messages
                "quantum_backend_selected": "已选择量子后端: {backend}",
                "classical_fallback": "使用经典后备优化",
                "optimization_converged": "优化在 {iterations} 次迭代后收敛",
                "makespan_optimized": "制作跨度优化为 {makespan} 小时",
                
                # Error messages
                "invalid_input": "提供了无效输入",
                "insufficient_resources": "任务分配资源不足",
                "optimization_failed": "优化收敛失败",
                "backend_unavailable": "量子后端不可用",
                
                # Status messages
                "ready": "就绪",
                "running": "运行中",
                "completed": "已完成",
                "failed": "失败",
                "cancelled": "已取消"
            }
        }
    
    def get_message(self, message_key: str, **kwargs) -> str:
        """Get localized message with optional formatting.
        
        Args:
            message_key: Key for the message to retrieve
            **kwargs: Format parameters for the message
            
        Returns:
            Localized and formatted message
        """
        lang_code = self.current_language.value
        
        # Get message from translations
        if lang_code in self.translations:
            message = self.translations[lang_code].get(message_key)
        else:
            # Fallback to English
            message = self.translations["en"].get(message_key)
        
        if message is None:
            # Fallback to message key if translation not found
            message = message_key.replace("_", " ").title()
        
        # Format message with parameters
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError):
            return message
    
    def set_language(self, language: Union[SupportedLanguage, str]):
        """Set the current language.
        
        Args:
            language: Language to set (SupportedLanguage enum or string code)
        """
        if isinstance(language, str):
            # Find matching language by code
            for lang in SupportedLanguage:
                if lang.value == language:
                    self.current_language = lang
                    break
            else:
                raise ValueError(f"Unsupported language code: {language}")
        else:
            self.current_language = language
    
    def get_current_language(self) -> SupportedLanguage:
        """Get the current language setting."""
        return self.current_language
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with metadata.
        
        Returns:
            List of language information dictionaries
        """
        languages = []
        for lang_code, locale_info in self.locale_info.items():
            languages.append({
                'code': lang_code,
                'name': locale_info.language_name,
                'is_rtl': locale_info.is_rtl
            })
        return languages
    
    def format_number(self, number: Union[int, float]) -> str:
        """Format number according to current locale.
        
        Args:
            number: Number to format
            
        Returns:
            Formatted number string
        """
        lang_code = self.current_language.value
        locale_info = self.locale_info.get(lang_code, self.locale_info["en"])
        
        # Simple number formatting
        if isinstance(number, float):
            formatted = f"{number:.2f}"
            # Replace decimal separator
            if locale_info.decimal_separator != ".":
                formatted = formatted.replace(".", locale_info.decimal_separator)
        else:
            formatted = str(number)
        
        # Add thousands separator for large numbers
        if abs(number) >= 1000 and locale_info.thousands_separator:
            parts = formatted.split(locale_info.decimal_separator)
            integer_part = parts[0]
            
            # Insert thousands separators
            formatted_integer = ""
            for i, digit in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    formatted_integer = locale_info.thousands_separator + formatted_integer
                formatted_integer = digit + formatted_integer
            
            if len(parts) > 1:
                formatted = formatted_integer + locale_info.decimal_separator + parts[1]
            else:
                formatted = formatted_integer
        
        return formatted
    
    def format_currency(self, amount: Union[int, float]) -> str:
        """Format currency according to current locale.
        
        Args:
            amount: Currency amount to format
            
        Returns:
            Formatted currency string
        """
        lang_code = self.current_language.value
        locale_info = self.locale_info.get(lang_code, self.locale_info["en"])
        
        formatted_number = self.format_number(amount)
        
        # Different currency formatting by locale
        if lang_code in ["de", "fr"]:
            return f"{formatted_number} {locale_info.currency_symbol}"
        elif lang_code == "ja":
            return f"{locale_info.currency_symbol}{formatted_number}"
        else:
            return f"{locale_info.currency_symbol}{formatted_number}"
    
    def is_rtl_language(self) -> bool:
        """Check if current language is right-to-left.
        
        Returns:
            True if current language is RTL
        """
        lang_code = self.current_language.value
        locale_info = self.locale_info.get(lang_code, self.locale_info["en"])
        return locale_info.is_rtl
    
    def get_locale_info(self) -> LocaleInfo:
        """Get locale information for current language.
        
        Returns:
            LocaleInfo object for current language
        """
        lang_code = self.current_language.value
        return self.locale_info.get(lang_code, self.locale_info["en"])


# Global instance for easy access
_i18n_manager = InternationalizationManager()


def set_language(language: Union[SupportedLanguage, str]):
    """Set the global language setting."""
    _i18n_manager.set_language(language)


def get_message(message_key: str, **kwargs) -> str:
    """Get localized message from global manager."""
    return _i18n_manager.get_message(message_key, **kwargs)


def format_number(number: Union[int, float]) -> str:
    """Format number using global locale settings."""
    return _i18n_manager.format_number(number)


def format_currency(amount: Union[int, float]) -> str:
    """Format currency using global locale settings."""
    return _i18n_manager.format_currency(amount)


def get_supported_languages() -> List[Dict[str, str]]:
    """Get supported languages from global manager."""
    return _i18n_manager.get_supported_languages()


def is_rtl_language() -> bool:
    """Check if current global language is RTL."""
    return _i18n_manager.is_rtl_language()