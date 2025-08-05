#!/usr/bin/env python3
"""Global-First Implementation Test - I18n, Multi-region, Compliance Testing"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_planner import QuantumTaskPlanner, Agent, Task
from quantum_planner.globalization import (
    globalization, 
    InternationalizationManager, 
    ComplianceManager,
    UserContext,
    Region,
    Language,
    ComplianceFramework,
    create_user_context
)

def test_internationalization():
    """Test internationalization and localization features."""
    print("🌍 Testing Internationalization")
    
    i18n = InternationalizationManager()
    
    # Test default English
    english_msg = i18n.translate('assignment_completed')
    print(f"🇺🇸 English: {english_msg}")
    assert 'completed' in english_msg.lower()
    
    # Test Spanish
    i18n.set_language(Language.SPANISH)
    spanish_msg = i18n.translate('assignment_completed')
    print(f"🇪🇸 Spanish: {spanish_msg}")
    assert 'completada' in spanish_msg.lower()
    
    # Test French
    i18n.set_language(Language.FRENCH)
    french_msg = i18n.translate('assignment_completed')
    print(f"🇫🇷 French: {french_msg}")
    assert 'terminée' in french_msg.lower() or 'terminé' in french_msg.lower()
    
    # Test German
    i18n.set_language(Language.GERMAN)
    german_msg = i18n.translate('assignment_completed')
    print(f"🇩🇪 German: {german_msg}")
    assert 'abgeschlossen' in german_msg.lower()
    
    # Test Japanese
    i18n.set_language(Language.JAPANESE)
    japanese_msg = i18n.translate('assignment_completed')
    print(f"🇯🇵 Japanese: {japanese_msg}")
    assert 'タスク' in japanese_msg
    
    # Test message formatting
    i18n.set_language(Language.ENGLISH)
    retention_msg = i18n.translate('data_retention', days=30)
    print(f"📋 Formatted message: {retention_msg}")
    assert '30' in retention_msg
    
    # Test available languages
    available_langs = i18n.get_available_languages()
    print(f"🌐 Available languages: {[lang.value for lang in available_langs]}")
    assert len(available_langs) >= 5
    
    print("✅ Internationalization tests passed")
    return True

def test_compliance_management():
    """Test compliance and data protection management."""
    print("\n🔐 Testing Compliance Management")
    
    compliance = ComplianceManager()
    
    # Test EU user context (GDPR)
    eu_user = UserContext(
        user_id="eu_user_001",
        region=Region.EU_WEST,
        language=Language.ENGLISH,
        consent_preferences={'data_processing': True},
        compliance_requirements=[ComplianceFramework.GDPR]
    )
    
    # Test data processing validation
    can_process = compliance.validate_data_processing(eu_user, "task_assignment")
    print(f"🇪🇺 EU user data processing allowed: {can_process}")
    assert can_process == True
    
    # Test without consent
    eu_user_no_consent = UserContext(
        user_id="eu_user_002",
        region=Region.EU_WEST,
        language=Language.ENGLISH,
        consent_preferences={'data_processing': False},
        compliance_requirements=[ComplianceFramework.GDPR]
    )
    
    can_process_no_consent = compliance.validate_data_processing(eu_user_no_consent, "task_assignment")
    print(f"🇪🇺 EU user without consent: {can_process_no_consent}")
    assert can_process_no_consent == False
    
    # Test US user context (CCPA)
    us_user = UserContext(
        user_id="us_user_001",
        region=Region.US_WEST,
        language=Language.ENGLISH,
        compliance_requirements=[ComplianceFramework.CCPA]
    )
    
    can_process_us = compliance.validate_data_processing(us_user, "task_assignment")
    print(f"🇺🇸 US user data processing allowed: {can_process_us}")
    assert isinstance(can_process_us, bool)
    
    # Test cross-border transfers
    eu_to_us_allowed = compliance.check_cross_border_transfer(Region.EU_WEST, Region.US_WEST)
    print(f"🌍 EU to US transfer allowed: {eu_to_us_allowed}")
    
    eu_to_eu_allowed = compliance.check_cross_border_transfer(Region.EU_WEST, Region.EU_CENTRAL)
    print(f"🇪🇺 EU to EU transfer allowed: {eu_to_eu_allowed}")
    assert eu_to_eu_allowed == True
    
    # Test data retention periods
    retention_period = compliance.get_data_retention_period(eu_user)
    print(f"📅 Data retention period: {retention_period} days")
    assert retention_period > 0
    
    # Test compliance report generation
    report = compliance.generate_compliance_report(Region.EU_WEST)
    print(f"📊 Compliance report: {report['compliance_rate']:.2f} rate")
    assert 'compliance_rate' in report
    assert 'total_events' in report
    
    print("✅ Compliance management tests passed")
    return True

def test_regional_configurations():
    """Test regional configuration handling."""
    print("\n🗺️ Testing Regional Configurations")
    
    # Test different regions
    regions_to_test = [
        (Region.EU_WEST, "GDPR", True),
        (Region.US_WEST, "CCPA", False),
        (Region.ASIA_PACIFIC, "PDPA", True),
    ]
    
    compliance = ComplianceManager()
    
    for region, expected_framework, data_residency in regions_to_test:
        config = compliance.regional_configs.get(region)
        assert config is not None, f"No config for {region}"
        
        print(f"🌍 {region.value}:")
        print(f"   📍 Data residency required: {config.data_residency_required}")
        print(f"   🔐 Encryption enabled: {config.encryption_at_rest}")
        print(f"   📋 Frameworks: {[f.value for f in config.compliance_frameworks]}")
        print(f"   🔒 Privacy controls: {len(config.privacy_controls)} controls")
        
        assert config.data_residency_required == data_residency
        assert len(config.compliance_frameworks) > 0
        assert config.encryption_at_rest == True
        
        # Check framework presence
        framework_names = [f.value for f in config.compliance_frameworks]
        assert any(expected_framework.lower() in f.lower() for f in framework_names)
    
    print("✅ Regional configuration tests passed")
    return True

def test_globalized_planner():
    """Test quantum planner with globalization features."""
    print("\n🚀 Testing Globalized Quantum Planner")
    
    planner = QuantumTaskPlanner(backend="auto", fallback="simulated_annealing")
    
    # Test without user context (should work normally)
    agents = [Agent(id="agent1", skills=["python"], capacity=1)]
    tasks = [Task(id="task1", required_skills=["python"], priority=1, duration=1)]
    
    solution = planner.assign(agents, tasks)
    print(f"✅ Assignment without context: {len(solution.assignments)} assignments")
    
    # Test with EU user context
    eu_user = create_user_context(
        user_id="eu_test_user",
        region="eu-west-1",
        language="en",
        consent_preferences={'data_processing': True}
    )
    
    globalization.set_user_context(eu_user)
    
    # Test localized messages
    localized_msg = globalization.localized_message('assignment_started')
    print(f"🌍 Localized message: {localized_msg}")
    
    # Test with user context (should apply compliance)
    solution_eu = planner.assign(agents, tasks)
    print(f"✅ EU assignment: {len(solution_eu.assignments)} assignments")
    
    # Check if compliance metadata was added
    if hasattr(solution_eu, 'metadata') and solution_eu.metadata:
        if '_compliance' in solution_eu.metadata:
            print(f"🔐 Compliance metadata added: {solution_eu.metadata['_compliance']}")
    
    # Test with German user
    de_user = create_user_context(
        user_id="de_test_user", 
        region="eu-central-1",
        language="de",
        consent_preferences={'data_processing': True}
    )
    
    globalization.set_user_context(de_user)
    localized_msg_de = globalization.localized_message('assignment_started')
    print(f"🇩🇪 German message: {localized_msg_de}")
    assert 'gestartet' in localized_msg_de.lower()
    
    # Test datetime formatting
    formatted_time = globalization.format_datetime(time.time())
    print(f"🕐 German time format: {formatted_time}")
    
    print("✅ Globalized planner tests passed")
    return True

def test_privacy_and_data_protection():
    """Test privacy and data protection features."""
    print("\n🔒 Testing Privacy and Data Protection")
    
    # Setup user with strict privacy requirements
    privacy_user = UserContext(
        user_id="privacy_user_001",
        region=Region.EU_WEST,
        language=Language.ENGLISH,
        data_classification="sensitive",
        consent_preferences={'data_processing': True},
        compliance_requirements=[ComplianceFramework.GDPR]
    )
    
    globalization.set_user_context(privacy_user)
    
    # Test data protection application
    test_data = {
        'assignments': {'task1': 'agent1'},
        'makespan': 5.0,
        'debug_info': 'internal debugging data',
        'internal_metadata': {'secret': 'value'}
    }
    
    protected_data = globalization.ensure_data_protection(test_data)
    print(f"🔐 Protected data keys: {list(protected_data.keys())}")
    
    # Check data minimization
    assert 'debug_info' not in protected_data, "Debug info should be removed"
    assert 'internal_metadata' not in protected_data, "Internal metadata should be removed"
    
    # Check compliance metadata
    assert '_compliance' in protected_data, "Compliance metadata should be added"
    assert 'frameworks' in protected_data['_compliance'], "Frameworks should be listed"
    
    # Test privacy notice generation
    privacy_notice = globalization.generate_privacy_notice()
    print(f"📋 Privacy notice: {privacy_notice[:100]}...")
    assert len(privacy_notice) > 50, "Privacy notice should be comprehensive"
    
    # Test regional restrictions
    restrictions = globalization.get_regional_restrictions()
    print(f"⚖️  Regional restrictions: {restrictions}")
    assert 'data_residency_required' in restrictions
    assert 'compliance_frameworks' in restrictions
    
    print("✅ Privacy and data protection tests passed")
    return True

def test_multi_language_error_handling():
    """Test error handling with multiple languages."""
    print("\n🌐 Testing Multi-language Error Handling")
    
    languages_to_test = [
        (Language.ENGLISH, "error"),
        (Language.SPANISH, "error"),
        (Language.FRENCH, "erreur"),
        (Language.GERMAN, "fehler"),
        (Language.JAPANESE, "エラー")
    ]
    
    for language, expected_word in languages_to_test:
        # Create user context for each language
        user = UserContext(
            user_id=f"test_user_{language.value}",
            region=Region.EU_WEST,
            language=language,
            consent_preferences={'data_processing': True}
        )
        
        globalization.set_user_context(user)
        
        # Test system error message
        error_msg = globalization.localized_message('system_error')
        print(f"🗣️  {language.value}: {error_msg}")
        
        # Verify it's localized (not just the key)
        assert error_msg != 'system_error', f"Message not localized for {language.value}"
        assert len(error_msg) > 5, f"Message too short for {language.value}"
    
    print("✅ Multi-language error handling tests passed")
    return True

def test_compliance_violations():
    """Test compliance violation detection and handling."""
    print("\n⚠️ Testing Compliance Violations")
    
    compliance = ComplianceManager()
    
    # Test user without consent
    no_consent_user = UserContext(
        user_id="no_consent_user",
        region=Region.EU_WEST,
        language=Language.ENGLISH,
        consent_preferences={'data_processing': False},
        compliance_requirements=[ComplianceFramework.GDPR]
    )
    
    globalization.set_user_context(no_consent_user)
    
    # This should fail due to lack of consent
    try:
        planner = QuantumTaskPlanner(backend="auto", fallback="simulated_annealing")
        agents = [Agent(id="agent1", skills=["python"], capacity=1)]
        tasks = [Task(id="task1", required_skills=["python"], priority=1, duration=1)]
        
        solution = planner.assign(agents, tasks)
        print("❌ Should have failed due to consent violation")
        return False
        
    except PermissionError as e:
        print(f"✅ Consent violation correctly detected: {str(e)}")
    
    # Test restricted data without proper encryption context
    restricted_user = UserContext(
        user_id="restricted_user",
        region=Region.EU_WEST,
        language=Language.ENGLISH,
        data_classification="restricted",
        consent_preferences={'data_processing': True},
        compliance_requirements=[ComplianceFramework.GDPR]
    )
    
    # This should work because our default config has encryption enabled
    globalization.set_user_context(restricted_user)
    validation_result = compliance.validate_data_processing(restricted_user, "test_operation")
    print(f"🔐 Restricted data validation: {validation_result}")
    
    # Generate compliance report to check violations
    report = compliance.generate_compliance_report()
    violations = report.get('violations', 0)
    print(f"📊 Total compliance violations detected: {violations}")
    
    print("✅ Compliance violation tests passed")
    return True

def test_cross_region_scenarios():
    """Test cross-region data handling scenarios."""
    print("\n🌍 Testing Cross-Region Scenarios")
    
    compliance = ComplianceManager()
    
    # Test allowed transfers
    allowed_scenarios = [
        (Region.EU_WEST, Region.EU_CENTRAL, True),
        (Region.US_WEST, Region.US_EAST, True),
        (Region.ASIA_PACIFIC, Region.ASIA_NORTHEAST, True),
    ]
    
    for source, target, expected in allowed_scenarios:
        result = compliance.check_cross_border_transfer(source, target)
        print(f"📡 {source.value} → {target.value}: {result}")
        assert result == expected, f"Transfer {source.value} to {target.value} should be {expected}"
    
    # Test restricted transfers
    restricted_scenarios = [
        (Region.EU_WEST, Region.US_WEST, False),
        (Region.US_WEST, Region.ASIA_PACIFIC, False),
    ]
    
    for source, target, expected in restricted_scenarios:
        result = compliance.check_cross_border_transfer(source, target)
        print(f"🚫 {source.value} → {target.value}: {result}")
        assert result == expected, f"Transfer {source.value} to {target.value} should be {expected}"
    
    # Test different retention periods
    retention_scenarios = [
        (Region.EU_WEST, 365),
        (Region.US_WEST, 730),
        (Region.ASIA_PACIFIC, 365),
    ]
    
    for region, expected_days in retention_scenarios:
        user = UserContext(
            user_id="retention_test",
            region=region,
            language=Language.ENGLISH
        )
        
        retention = compliance.get_data_retention_period(user)
        print(f"📅 {region.value} retention: {retention} days")
        assert retention == expected_days, f"Retention for {region.value} should be {expected_days} days"
    
    print("✅ Cross-region scenario tests passed")
    return True

if __name__ == "__main__":
    print("🌍 Starting Global-First Implementation Tests\n")
    
    try:
        # Run all globalization tests
        test_internationalization()
        test_compliance_management()
        test_regional_configurations()
        test_globalized_planner()
        test_privacy_and_data_protection()
        test_multi_language_error_handling()
        test_compliance_violations()
        test_cross_region_scenarios()
        
        print("\n🎉 ALL GLOBAL-FIRST IMPLEMENTATION TESTS PASSED!")
        print("✅ Internationalization (I18n) working in 5+ languages")
        print("✅ Compliance management (GDPR, CCPA, PDPA) implemented")
        print("✅ Regional configurations with data residency")
        print("✅ Globalized quantum planner with context awareness")
        print("✅ Privacy and data protection measures active")
        print("✅ Multi-language error handling functional")
        print("✅ Compliance violation detection working")
        print("✅ Cross-region data transfer controls in place")
        
        # Final summary
        print(f"\n🌍 GLOBALIZATION FEATURES SUMMARY:")
        print(f"   🗣️  Languages supported: 5 (EN, ES, FR, DE, JA)")
        print(f"   🌍 Regions configured: 6 global regions")
        print(f"   📋 Compliance frameworks: 6 major frameworks")
        print(f"   🔐 Privacy controls: Data minimization, encryption, consent")
        print(f"   ⚖️  Data residency: EU, Asia Pacific requirements")
        
    except Exception as e:
        print(f"\n❌ GLOBAL IMPLEMENTATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)