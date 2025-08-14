"""Global-first implementation with multi-region support, i18n, and compliance."""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported global regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"


class Language(Enum):
    """Supported languages."""
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    JA = "ja"
    ZH = "zh"
    PT = "pt"
    RU = "ru"
    
    # Backward compatibility aliases
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"          # General Data Protection Regulation (EU)
    CCPA = "ccpa"          # California Consumer Privacy Act (US)
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore)
    SOC2 = "soc2"          # Service Organization Control 2
    ISO27001 = "iso27001"  # Information Security Management
    HIPAA = "hipaa"        # Health Insurance Portability Act (US)


@dataclass
class RegionalConfig:
    """Configuration for a specific region."""
    region: Region
    data_residency_required: bool = False
    encryption_in_transit: bool = True
    encryption_at_rest: bool = True
    data_retention_days: int = 365
    allowed_data_transfers: List[Region] = field(default_factory=list)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    privacy_controls: Dict[str, bool] = field(default_factory=dict)


@dataclass
class UserContext:
    """User context for localization and compliance."""
    user_id: str
    region: Region
    language: Language
    timezone: str = "UTC"
    consent_preferences: Dict[str, bool] = field(default_factory=dict)
    data_classification: str = "standard"  # standard, sensitive, restricted
    compliance_requirements: List[ComplianceFramework] = field(default_factory=list)


class InternationalizationManager:
    """Manages internationalization and localization."""
    
    def __init__(self):
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = Language.EN
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files."""
        # Default English translations
        self.translations[Language.EN.value] = {
            'assignment_started': 'Task assignment started',
            'assignment_completed': 'Task assignment completed',
            'optimization_started': 'Optimization started',
            'optimization_completed': 'Optimization completed in {duration}',
            'optimization_backend': 'Optimization backend',
            'solution_found': 'Solution found',
            'no_solution': 'No feasible solution found',
            'skill_mismatch': 'Skill requirements do not match available agents',
            'capacity_exceeded': 'Agent capacity exceeded',
            'invalid_input': 'Invalid input provided',
            'system_error': 'System error occurred',
            'performance_warning': 'Performance degradation detected',
            'cache_hit': 'Using cached solution',
            'fallback_used': 'Fallback backend used',
            'validation_failed': 'Solution validation failed',
            'concurrent_limit': 'Concurrent operation limit reached',
            'data_processed': 'Data processed successfully',
            'privacy_notice': 'Your data is processed according to our privacy policy',
            'consent_required': 'User consent required for data processing',
            'data_retention': 'Data will be retained for {days} days',
            'cross_border_transfer': 'Data may be transferred across regions',
            'encryption_enabled': 'Data encryption is enabled',
            'compliance_check': 'Compliance validation completed',
            'audit_log': 'Action logged for audit purposes',
        }
        
        # Spanish translations
        self.translations[Language.ES.value] = {
            'assignment_started': 'Asignación de tareas iniciada',
            'assignment_completed': 'Asignación de tareas completada',
            'optimization_started': 'Optimización iniciada',
            'optimization_completed': 'Optimización completada en {duration}',
            'optimization_backend': 'Backend de optimización',
            'solution_found': 'Solución encontrada',
            'no_solution': 'No se encontró una solución factible',
            'skill_mismatch': 'Los requisitos de habilidades no coinciden con los agentes disponibles',
            'capacity_exceeded': 'Capacidad del agente excedida',
            'invalid_input': 'Entrada inválida proporcionada',
            'system_error': 'Error del sistema ocurrido',
            'performance_warning': 'Degradación de rendimiento detectada',
            'cache_hit': 'Usando solución en caché',
            'fallback_used': 'Backend de respaldo utilizado',
            'validation_failed': 'Validación de solución falló',
            'concurrent_limit': 'Límite de operación concurrente alcanzado',
            'data_processed': 'Datos procesados exitosamente',
            'privacy_notice': 'Sus datos se procesan según nuestra política de privacidad',
            'consent_required': 'Se requiere consentimiento del usuario para el procesamiento de datos',
            'data_retention': 'Los datos se conservarán durante {days} días',
            'cross_border_transfer': 'Los datos pueden transferirse entre regiones',
            'encryption_enabled': 'El cifrado de datos está habilitado',
            'compliance_check': 'Validación de cumplimiento completada',
            'audit_log': 'Acción registrada para fines de auditoría',
        }
        
        # French translations
        self.translations[Language.FR.value] = {
            'assignment_started': 'Attribution des tâches commencée',
            'assignment_completed': 'Attribution des tâches terminée',
            'optimization_started': 'Optimisation commencée',
            'optimization_completed': 'Optimisation terminée en {duration}',
            'optimization_backend': 'Backend d\'optimisation',
            'solution_found': 'Solution trouvée',
            'no_solution': 'Aucune solution réalisable trouvée',
            'skill_mismatch': 'Les exigences de compétences ne correspondent pas aux agents disponibles',
            'capacity_exceeded': 'Capacité de l\'agent dépassée',
            'invalid_input': 'Entrée invalide fournie',
            'system_error': 'Erreur système survenue',
            'performance_warning': 'Dégradation des performances détectée',
            'cache_hit': 'Utilisation de la solution mise en cache',
            'fallback_used': 'Backend de secours utilisé',
            'validation_failed': 'Échec de la validation de la solution',
            'concurrent_limit': 'Limite d\'opération simultanée atteinte',
            'data_processed': 'Données traitées avec succès',
            'privacy_notice': 'Vos données sont traitées selon notre politique de confidentialité',
            'consent_required': 'Consentement de l\'utilisateur requis pour le traitement des données',
            'data_retention': 'Les données seront conservées pendant {days} jours',
            'cross_border_transfer': 'Les données peuvent être transférées entre régions',
            'encryption_enabled': 'Le chiffrement des données est activé',
            'compliance_check': 'Validation de conformité terminée',
            'audit_log': 'Action enregistrée à des fins d\'audit',
        }
        
        # German translations
        self.translations[Language.DE.value] = {
            'assignment_started': 'Aufgabenzuweisung gestartet',
            'assignment_completed': 'Aufgabenzuweisung abgeschlossen',
            'optimization_started': 'Optimierung gestartet',
            'optimization_completed': 'Optimierung abgeschlossen in {duration}',
            'optimization_backend': 'Optimierungs-Backend',
            'solution_found': 'Lösung gefunden',
            'no_solution': 'Keine durchführbare Lösung gefunden',
            'skill_mismatch': 'Fertigkeitsanforderungen stimmen nicht mit verfügbaren Agenten überein',
            'capacity_exceeded': 'Agentenkapazität überschritten',
            'invalid_input': 'Ungültige Eingabe bereitgestellt',
            'system_error': 'Systemfehler aufgetreten',
            'performance_warning': 'Leistungsverschlechterung erkannt',
            'cache_hit': 'Verwende zwischengespeicherte Lösung',
            'fallback_used': 'Fallback-Backend verwendet',
            'validation_failed': 'Lösungsvalidierung fehlgeschlagen',
            'concurrent_limit': 'Grenze für gleichzeitige Operationen erreicht',
            'data_processed': 'Daten erfolgreich verarbeitet',
            'privacy_notice': 'Ihre Daten werden gemäß unserer Datenschutzrichtlinie verarbeitet',
            'consent_required': 'Benutzereinwilligung erforderlich für Datenverarbeitung',
            'data_retention': 'Daten werden für {days} Tage aufbewahrt',
            'cross_border_transfer': 'Daten können zwischen Regionen übertragen werden',
            'encryption_enabled': 'Datenverschlüsselung ist aktiviert',
            'compliance_check': 'Compliance-Validierung abgeschlossen',
            'audit_log': 'Aktion zu Prüfzwecken protokolliert',
        }
        
        # Japanese translations
        self.translations[Language.JA.value] = {
            'assignment_started': 'タスク割り当てが開始されました',
            'assignment_completed': 'タスク割り当てが完了しました',
            'optimization_started': '最適化が開始されました',
            'optimization_completed': '{duration}で最適化が完了しました',
            'optimization_backend': '最適化バックエンド',
            'solution_found': '解決策が見つかりました',
            'no_solution': '実行可能な解決策が見つかりません',
            'skill_mismatch': 'スキル要件が利用可能なエージェントと一致しません',
            'capacity_exceeded': 'エージェントの容量を超過しました',
            'invalid_input': '無効な入力が提供されました',
            'system_error': 'システムエラーが発生しました',
            'performance_warning': 'パフォーマンスの低下が検出されました',
            'cache_hit': 'キャッシュされた解決策を使用',
            'fallback_used': 'フォールバックバックエンドを使用',
            'validation_failed': '解決策の検証に失敗しました',
            'concurrent_limit': '同時実行の制限に達しました',
            'data_processed': 'データが正常に処理されました',
            'privacy_notice': 'お客様のデータはプライバシーポリシーに従って処理されます',
            'consent_required': 'データ処理にはユーザーの同意が必要です',
            'data_retention': 'データは{days}日間保持されます',
            'cross_border_transfer': 'データは地域間で転送される場合があります',
            'encryption_enabled': 'データ暗号化が有効になっています',
            'compliance_check': 'コンプライアンス検証が完了しました',
            'audit_log': '監査目的でアクションがログに記録されました',
        }
        
        # Chinese translations
        self.translations[Language.ZH.value] = {
            'assignment_started': '任务分配已开始',
            'assignment_completed': '任务分配已完成',
            'optimization_started': '优化已开始',
            'optimization_completed': '优化在{duration}内完成',
            'optimization_backend': '优化后端',
            'solution_found': '找到解决方案',
            'no_solution': '未找到可行解决方案',
            'skill_mismatch': '技能要求与可用代理不匹配',
            'capacity_exceeded': '代理容量超出',
            'invalid_input': '提供的输入无效',
            'system_error': '系统错误发生',
            'performance_warning': '检测到性能下降',
            'cache_hit': '使用缓存解决方案',
            'fallback_used': '使用备用后端',
            'validation_failed': '解决方案验证失败',
            'concurrent_limit': '达到并发操作限制',
            'data_processed': '数据处理成功',
            'privacy_notice': '您的数据根据我们的隐私政策进行处理',
            'consent_required': '数据处理需要用户同意',
            'data_retention': '数据将保留{days}天',
            'cross_border_transfer': '数据可能在地区间传输',
            'encryption_enabled': '数据加密已启用',
            'compliance_check': '合规验证已完成',
            'audit_log': '操作已记录用于审计目的',
        }
    
    def set_language(self, language: Language):
        """Set the current language."""
        self.current_language = language
        logger.info(f"Language set to {language.value}")
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a message key to the current language."""
        translations = self.translations.get(self.current_language.value, {})
        message = translations.get(key, key)  # Fallback to key if not found
        
        # Handle string formatting
        if kwargs:
            try:
                message = message.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Translation formatting error for key '{key}': {e}")
        
        return message
    
    def get_available_languages(self) -> List[Language]:
        """Get list of available languages."""
        return [Language(lang) for lang in self.translations.keys()]


class ComplianceManager:
    """Manages compliance and data protection requirements."""
    
    def __init__(self):
        self.regional_configs: Dict[Region, RegionalConfig] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default regional compliance configurations."""
        
        # EU region - GDPR compliance
        self.regional_configs[Region.EU_WEST] = RegionalConfig(
            region=Region.EU_WEST,
            data_residency_required=True,
            encryption_in_transit=True,
            encryption_at_rest=True,
            data_retention_days=365,
            allowed_data_transfers=[Region.EU_CENTRAL],
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
            privacy_controls={
                'explicit_consent': True,
                'right_to_erasure': True,
                'data_portability': True,
                'privacy_by_design': True,
                'data_minimization': True
            }
        )
        
        # US region - CCPA compliance
        self.regional_configs[Region.US_WEST] = RegionalConfig(
            region=Region.US_WEST,
            data_residency_required=False,
            encryption_in_transit=True,
            encryption_at_rest=True,
            data_retention_days=730,
            allowed_data_transfers=[Region.US_EAST],
            compliance_frameworks=[ComplianceFramework.CCPA, ComplianceFramework.SOC2],
            privacy_controls={
                'opt_out_right': True,
                'data_transparency': True,
                'non_discrimination': True,
                'secure_deletion': True
            }
        )
        
        # Asia Pacific - PDPA compliance
        self.regional_configs[Region.ASIA_PACIFIC] = RegionalConfig(
            region=Region.ASIA_PACIFIC,
            data_residency_required=True,
            encryption_in_transit=True,
            encryption_at_rest=True,
            data_retention_days=365,
            allowed_data_transfers=[Region.ASIA_NORTHEAST],
            compliance_frameworks=[ComplianceFramework.PDPA, ComplianceFramework.ISO27001],
            privacy_controls={
                'consent_required': True,
                'purpose_limitation': True,
                'accuracy_requirement': True,
                'security_arrangements': True
            }
        )
    
    def validate_data_processing(self, user_context: UserContext, operation: str) -> bool:
        """Validate if data processing is allowed for user context."""
        config = self.regional_configs.get(user_context.region)
        if not config:
            logger.warning(f"No compliance config for region {user_context.region}")
            return False
        
        # Check consent requirements
        if ComplianceFramework.GDPR in config.compliance_frameworks:
            if not user_context.consent_preferences.get('data_processing', False):
                self._log_compliance_event(user_context, operation, 'consent_missing')
                return False
        
        # Check data classification restrictions
        if user_context.data_classification == 'restricted':
            if not all([config.encryption_in_transit, config.encryption_at_rest]):
                self._log_compliance_event(user_context, operation, 'encryption_required')
                return False
        
        self._log_compliance_event(user_context, operation, 'validated')
        return True
    
    def check_cross_border_transfer(self, source_region: Region, target_region: Region) -> bool:
        """Check if cross-border data transfer is allowed."""
        source_config = self.regional_configs.get(source_region)
        if not source_config:
            return False
        
        # Check if target region is in allowed transfers
        if target_region not in source_config.allowed_data_transfers:
            logger.warning(f"Cross-border transfer not allowed: {source_region} -> {target_region}")
            return False
        
        return True
    
    def get_data_retention_period(self, user_context: UserContext) -> int:
        """Get data retention period for user context."""
        config = self.regional_configs.get(user_context.region)
        if not config:
            return 365  # Default 1 year
        
        return config.data_retention_days
    
    def _log_compliance_event(self, user_context: UserContext, operation: str, event_type: str):
        """Log compliance event for audit purposes."""
        event = {
            'timestamp': time.time(),
            'user_id': user_context.user_id,
            'region': user_context.region.value,
            'operation': operation,
            'event_type': event_type,
            'compliance_frameworks': [f.value for f in user_context.compliance_requirements]
        }
        
        self.audit_log.append(event)
        
        # Keep only last 10000 events
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
    
    def generate_compliance_report(self, region: Optional[Region] = None) -> Dict[str, Any]:
        """Generate compliance report."""
        if region:
            events = [e for e in self.audit_log if e['region'] == region.value]
        else:
            events = self.audit_log
        
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Calculate compliance metrics
        total_events = len(events)
        validated_events = event_counts.get('validated', 0)
        compliance_rate = validated_events / total_events if total_events > 0 else 1.0
        
        return {
            'region': region.value if region else 'all',
            'reporting_period': {
                'start': min(e['timestamp'] for e in events) if events else time.time(),
                'end': max(e['timestamp'] for e in events) if events else time.time(),
            },
            'total_events': total_events,
            'compliance_rate': compliance_rate,
            'event_breakdown': event_counts,
            'violations': event_counts.get('consent_missing', 0) + event_counts.get('encryption_required', 0),
            'generated_at': time.time()
        }


class GlobalizationManager:
    """Main manager for globalization features."""
    
    def __init__(self):
        self.i18n = InternationalizationManager()
        self.compliance = ComplianceManager()
        self.current_user_context: Optional[UserContext] = None
        self.current_region = Region.US_EAST  # Default region
        self.current_language = Language.EN  # Default language
        self.data_records: List[Dict[str, Any]] = []  # Track processed data
        self.compliance_records: List[Dict[str, Any]] = []  # Track compliance records
    
    def set_user_context(self, user_context: UserContext):
        """Set current user context for operations."""
        self.current_user_context = user_context
        self.i18n.set_language(user_context.language)
        logger.info(f"User context set: {user_context.user_id} in {user_context.region.value}")
    
    def localized_message(self, key: str, **kwargs) -> str:
        """Get localized message for current user context."""
        return self.i18n.translate(key, **kwargs)
    
    def validate_operation(self, operation: str) -> bool:
        """Validate if operation is allowed for current user context."""
        if not self.current_user_context:
            return True  # Allow if no context set
        
        return self.compliance.validate_data_processing(self.current_user_context, operation)
    
    def format_datetime(self, timestamp: float) -> str:
        """Format datetime according to user's locale."""
        if not self.current_user_context:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        
        # Simple timezone handling (in production, use proper timezone libraries)
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        # Format based on language
        if self.current_user_context.language == Language.GERMAN:
            return dt.strftime("%d.%m.%Y %H:%M:%S")
        elif self.current_user_context.language == Language.JAPANESE:
            return dt.strftime("%Y年%m月%d日 %H時%M分")
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def get_regional_restrictions(self) -> Dict[str, Any]:
        """Get regional restrictions for current user."""
        if not self.current_user_context:
            return {}
        
        config = self.compliance.regional_configs.get(self.current_user_context.region)
        if not config:
            return {}
        
        return {
            'data_residency_required': config.data_residency_required,
            'encryption_required': config.encryption_in_transit and config.encryption_at_rest,
            'data_retention_days': config.data_retention_days,
            'compliance_frameworks': [f.value for f in config.compliance_frameworks],
            'privacy_controls': config.privacy_controls
        }
    
    def ensure_data_protection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure data protection measures are applied."""
        if not self.current_user_context:
            return data
        
        config = self.compliance.regional_configs.get(self.current_user_context.region)
        if not config:
            return data
        
        protected_data = data.copy()
        
        # Apply data minimization if required
        if config.privacy_controls.get('data_minimization', False):
            # Remove unnecessary fields (simplified example)
            fields_to_minimize = ['debug_info', 'internal_metadata']
            for field in fields_to_minimize:
                protected_data.pop(field, None)
        
        # Add encryption marker
        if config.encryption_at_rest:
            protected_data['_encryption_applied'] = True
        
        # Add compliance metadata
        protected_data['_compliance'] = {
            'frameworks': [f.value for f in config.compliance_frameworks],
            'region': config.region.value,
            'data_classification': self.current_user_context.data_classification
        }
        
        return protected_data
    
    def generate_privacy_notice(self) -> str:
        """Generate privacy notice for current user."""
        if not self.current_user_context:
            return self.i18n.translate('privacy_notice')
        
        config = self.compliance.regional_configs.get(self.current_user_context.region)
        if not config:
            return self.i18n.translate('privacy_notice')
        
        notice_parts = [
            self.i18n.translate('privacy_notice'),
            self.i18n.translate('data_retention', days=config.data_retention_days)
        ]
        
        if config.encryption_at_rest:
            notice_parts.append(self.i18n.translate('encryption_enabled'))
        
        if len(config.allowed_data_transfers) > 0:
            notice_parts.append(self.i18n.translate('cross_border_transfer'))
        
        return ' '.join(notice_parts)
    
    def set_region(self, region: Region) -> None:
        """Set current region."""
        self.current_region = region
        logger.info(f"Region set to {region.value}")
    
    def set_language(self, language: Language) -> None:
        """Set current language."""
        self.current_language = language
        self.i18n.set_language(language)
        logger.info(f"Language set to {language.value}")
    
    def get_regional_config(self, region: Region) -> RegionalConfig:
        """Get configuration for a specific region."""
        return self.compliance.regional_configs.get(region, RegionalConfig(region=region))
    
    def get_quantum_backends_for_region(self, region: Region) -> List[str]:
        """Get available quantum backends for a region."""
        # Simulate quantum backend availability by region
        backend_map = {
            Region.US_EAST: ["ibm_quantum_us", "aws_braket_us_east"],
            Region.US_WEST: ["google_quantum_us", "aws_braket_us_west"],
            Region.EU_WEST: ["ibm_quantum_eu", "atos_quantum_eu"],
            Region.EU_CENTRAL: ["cambridge_quantum_eu", "pasqal_eu"],
            Region.ASIA_PACIFIC: ["rigetti_apac", "aws_braket_apac"],
            Region.ASIA_NORTHEAST: ["ibm_quantum_jp", "ntt_quantum_jp"]
        }
        return backend_map.get(region, [])
    
    def get_message(self, key: str, **kwargs) -> str:
        """Get localized message."""
        return self.i18n.translate(key, **kwargs)
    
    def check_compliance_requirements(self, user_location: str, data_types: List[str]) -> Dict[str, Any]:
        """Check compliance requirements for user location and data types."""
        # Map location codes to regions and frameworks
        location_mapping = {
            "EU-DE": (Region.EU_WEST, [ComplianceFramework.GDPR]),
            "EU-FR": (Region.EU_WEST, [ComplianceFramework.GDPR]),
            "US-CA": (Region.US_WEST, [ComplianceFramework.CCPA]),
            "US-NY": (Region.US_EAST, [ComplianceFramework.CCPA]),
            "SG": (Region.ASIA_PACIFIC, [ComplianceFramework.PDPA]),
            "JP": (Region.ASIA_NORTHEAST, [ComplianceFramework.PDPA])
        }
        
        region, frameworks = location_mapping.get(user_location, (Region.US_EAST, []))
        
        # Check if personal data requires consent
        consent_required = "personal_data" in data_types and any(
            framework in [ComplianceFramework.GDPR, ComplianceFramework.PDPA] 
            for framework in frameworks
        )
        
        # Define user rights based on framework
        user_rights = []
        if ComplianceFramework.GDPR in frameworks:
            user_rights = ["access", "deletion", "rectification", "portability", "restriction"]
        elif ComplianceFramework.CCPA in frameworks:
            user_rights = ["access", "deletion", "opt_out", "non_discrimination"]
        elif ComplianceFramework.PDPA in frameworks:
            user_rights = ["access", "correction", "deletion"]
        
        return {
            "applicable_frameworks": [f.value for f in frameworks],
            "consent_required": consent_required,
            "user_rights": user_rights,
            "data_residency_required": region in [Region.EU_WEST, Region.EU_CENTRAL, Region.ASIA_PACIFIC],
            "encryption_required": True,
            "region": region.value
        }
    
    def process_data_with_compliance(
        self, 
        user_id: str, 
        data: Dict[str, Any], 
        purpose: str, 
        user_location: str,
        consent_given: bool = False
    ) -> Dict[str, Any]:
        """Process data with compliance requirements."""
        # Check compliance requirements
        requirements = self.check_compliance_requirements(user_location, list(data.keys()))
        
        if requirements["consent_required"] and not consent_given:
            raise ValueError("User consent required for data processing")
        
        # Determine processing region based on data residency requirements
        if requirements["data_residency_required"]:
            if user_location.startswith("EU"):
                processing_region = Region.EU_WEST
            elif user_location == "SG":
                processing_region = Region.ASIA_PACIFIC
            elif user_location == "JP":
                processing_region = Region.ASIA_NORTHEAST
            else:
                processing_region = self.current_region
        else:
            processing_region = self.current_region
        
        # Process data with encryption if required
        processed_data = data.copy()
        if requirements["encryption_required"] and "personal_data" in data:
            # Simulate encryption
            processed_data["personal_data_encrypted"] = hashlib.sha256(
                str(data["personal_data"]).encode()
            ).hexdigest()[:16]
            del processed_data["personal_data"]
        
        # Record compliance event
        compliance_record = {
            "timestamp": time.time(),
            "user_id": user_id,
            "purpose": purpose,
            "user_location": user_location,
            "processing_region": processing_region.value,
            "consent_given": consent_given,
            "frameworks": requirements["applicable_frameworks"],
            "data_types": list(data.keys())
        }
        self.compliance_records.append(compliance_record)
        
        # Record data processing
        data_record = {
            "user_id": user_id,
            "timestamp": time.time(),
            "data_types": list(data.keys()),
            "region": processing_region.value,
            "purpose": purpose
        }
        self.data_records.append(data_record)
        
        return {
            "processed_data": processed_data,
            "region": processing_region.value,
            "compliance_frameworks": requirements["applicable_frameworks"],
            "encryption_applied": requirements["encryption_required"]
        }
    
    def get_user_data_report(self, user_id: str) -> Dict[str, Any]:
        """Generate user data report for data access requests."""
        user_records = [r for r in self.data_records if r["user_id"] == user_id]
        user_compliance = [r for r in self.compliance_records if r["user_id"] == user_id]
        
        # Get user rights based on most recent compliance record
        user_rights = []
        if user_compliance:
            latest_record = max(user_compliance, key=lambda x: x["timestamp"])
            frameworks = latest_record["frameworks"]
            if "gdpr" in frameworks:
                user_rights = ["access", "deletion", "rectification", "portability"]
            elif "ccpa" in frameworks:
                user_rights = ["access", "deletion", "opt_out"]
            elif "pdpa" in frameworks:
                user_rights = ["access", "correction", "deletion"]
        
        return {
            "user_id": user_id,
            "report_generated_at": time.time(),
            "total_records": len(user_records),
            "data_processing_activities": [
                {
                    "timestamp": r["timestamp"],
                    "purpose": r["purpose"],
                    "data_types": r["data_types"],
                    "region": r["region"]
                }
                for r in user_records
            ],
            "user_rights": user_rights,
            "compliance_records": len(user_compliance)
        }
    
    def delete_user_data(self, user_id: str, verification_token: str) -> Dict[str, Any]:
        """Delete user data (right to be forgotten)."""
        # Simple verification (in production, use proper token validation)
        expected_token = hashlib.sha256(f"delete:{user_id}".encode()).hexdigest()[:16]
        if verification_token != expected_token:
            raise ValueError("Invalid verification token")
        
        # Count records before deletion
        initial_data_records = len(self.data_records)
        initial_compliance_records = len(self.compliance_records)
        
        # Remove user data records
        self.data_records = [r for r in self.data_records if r["user_id"] != user_id]
        self.compliance_records = [r for r in self.compliance_records if r["user_id"] != user_id]
        
        # Calculate deleted records
        deleted_data = initial_data_records - len(self.data_records)
        deleted_compliance = initial_compliance_records - len(self.compliance_records)
        
        return {
            "user_id": user_id,
            "deleted_at": time.time(),
            "records_deleted": deleted_data + deleted_compliance,
            "data_records_deleted": deleted_data,
            "compliance_records_deleted": deleted_compliance,
            "status": "completed"
        }
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard."""
        # Calculate metrics
        total_records = len(self.compliance_records)
        unique_users = len(set(r["user_id"] for r in self.compliance_records))
        active_regions = len(set(r["processing_region"] for r in self.compliance_records))
        
        # Calculate consent rate
        consent_given = sum(1 for r in self.compliance_records if r["consent_given"])
        consent_rate = consent_given / total_records if total_records > 0 else 1.0
        
        # Regional distribution
        regional_dist = {}
        for record in self.compliance_records:
            region = record["processing_region"]
            regional_dist[region] = regional_dist.get(region, 0) + 1
        
        # Framework distribution
        framework_dist = {}
        for record in self.compliance_records:
            for framework in record["frameworks"]:
                framework_dist[framework] = framework_dist.get(framework, 0) + 1
        
        return {
            "overview": {
                "total_compliance_records": total_records,
                "unique_users": unique_users,
                "active_regions": active_regions,
                "report_generated_at": time.time()
            },
            "consent_metrics": {
                "consent_rate": consent_rate,
                "total_consents": consent_given,
                "total_requests": total_records
            },
            "regional_distribution": regional_dist,
            "framework_distribution": framework_dist,
            "data_processing_summary": {
                "total_data_records": len(self.data_records),
                "active_users": len(set(r["user_id"] for r in self.data_records))
            }
        }


# Global instance
globalization = GlobalizationManager()


def with_globalization(func):
    """Decorator to add globalization context to functions."""
    def wrapper(*args, **kwargs):
        # Extract globalization parameters from kwargs
        region = kwargs.pop('region', None)
        language = kwargs.pop('language', None)
        
        # Store original state
        original_region = globalization.current_region
        original_language = globalization.current_language
        
        try:
            # Set temporary globalization context if provided
            if region:
                globalization.set_region(region)
            if language:
                globalization.set_language(language)
            
            # Validate operation if user context is set
            if not globalization.validate_operation(func.__name__):
                raise PermissionError(globalization.localized_message('consent_required'))
            
            result = func(*args, **kwargs)
            
            # Apply data protection to result if it's a dict
            if isinstance(result, dict):
                result = globalization.ensure_data_protection(result)
            
            return result
            
        except Exception as e:
            # Localize error messages
            localized_error = globalization.localized_message('system_error')
            logger.error(f"{localized_error}: {e}")
            raise
        
        finally:
            # Restore original state
            globalization.set_region(original_region)
            globalization.set_language(original_language)
    
    return wrapper


def create_user_context(user_id: str, region: str, language: str, **kwargs) -> UserContext:
    """Convenience function to create user context."""
    return UserContext(
        user_id=user_id,
        region=Region(region),
        language=Language(language),
        **kwargs
    )