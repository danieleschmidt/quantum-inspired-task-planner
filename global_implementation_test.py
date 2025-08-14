#!/usr/bin/env python3
"""Global-First Implementation Test - Multi-region, I18n, and Compliance."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import logging
from typing import Dict, Any
from quantum_planner import QuantumTaskPlanner, Agent, Task
from quantum_planner.globalization import (
    globalization, Region, Language, ComplianceFramework, with_globalization
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multi_region_support():
    """Test multi-region deployment capabilities."""
    logger.info("=== Testing Multi-Region Support ===")
    
    try:
        # Test region switching
        original_region = globalization.current_region
        
        for region in [Region.US_EAST, Region.EU_WEST, Region.ASIA_PACIFIC]:
            globalization.set_region(region)
            
            if globalization.current_region == region:
                logger.info(f"✅ Successfully switched to region: {region.value}")
            else:
                logger.error(f"❌ Failed to switch to region: {region.value}")
                return False
            
            # Test region-specific configuration
            config = globalization.get_regional_config(region)
            if config.region == region:
                logger.info(f"✅ Regional config loaded for {region.value}")
            else:
                logger.error(f"❌ Regional config mismatch for {region.value}")
                return False
            
            # Test quantum backends for region
            backends = globalization.get_quantum_backends_for_region(region)
            if backends:
                logger.info(f"✅ Quantum backends available in {region.value}: {backends}")
            else:
                logger.warning(f"⚠️ No quantum backends in {region.value}")
        
        # Restore original region
        globalization.set_region(original_region)
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-region test failed: {e}")
        return False


def test_internationalization():
    """Test internationalization (i18n) features."""
    logger.info("=== Testing Internationalization (i18n) ===")
    
    try:
        # Test language switching
        original_language = globalization.current_language
        
        test_languages = [Language.EN, Language.ES, Language.FR, Language.DE, Language.JA, Language.ZH]
        
        for language in test_languages:
            globalization.set_language(language)
            
            if globalization.current_language == language:
                logger.info(f"✅ Successfully switched to language: {language.value}")
            else:
                logger.error(f"❌ Failed to switch to language: {language.value}")
                return False
            
            # Test localized messages
            test_keys = ["optimization_started", "optimization_completed", "invalid_input"]
            
            for key in test_keys:
                message = globalization.get_message(key)
                if message and message != key:  # Should be translated, not fallback to key
                    logger.info(f"✅ {language.value}: '{key}' -> '{message}'")
                else:
                    logger.warning(f"⚠️ {language.value}: No translation for '{key}'")
        
        # Test message formatting
        globalization.set_language(Language.EN)
        formatted_msg = globalization.get_message("optimization_completed", duration="2.5s")
        if "optimization" in formatted_msg.lower():
            logger.info("✅ Message formatting works")
        else:
            logger.error("❌ Message formatting failed")
            return False
        
        # Restore original language
        globalization.set_language(original_language)
        return True
        
    except Exception as e:
        logger.error(f"❌ Internationalization test failed: {e}")
        return False


def test_compliance_frameworks():
    """Test compliance framework support (GDPR, CCPA, PDPA)."""
    logger.info("=== Testing Compliance Frameworks ===")
    
    try:
        # Test GDPR compliance (EU users)
        gdpr_requirements = globalization.check_compliance_requirements(
            user_location="EU-DE",
            data_types=["personal_data", "usage_metrics"]
        )
        
        if ComplianceFramework.GDPR.value in gdpr_requirements["applicable_frameworks"]:
            logger.info("✅ GDPR compliance detected for EU users")
        else:
            logger.error("❌ GDPR compliance not detected for EU users")
            return False
        
        if gdpr_requirements["consent_required"]:
            logger.info("✅ Consent requirement identified for personal data")
        else:
            logger.error("❌ Consent requirement not identified")
            return False
        
        # Test CCPA compliance (California users)
        ccpa_requirements = globalization.check_compliance_requirements(
            user_location="US-CA",
            data_types=["personal_data"]
        )
        
        if ComplianceFramework.CCPA.value in ccpa_requirements["applicable_frameworks"]:
            logger.info("✅ CCPA compliance detected for California users")
        else:
            logger.error("❌ CCPA compliance not detected for California users")
            return False
        
        # Test PDPA compliance (Singapore users)
        pdpa_requirements = globalization.check_compliance_requirements(
            user_location="SG",
            data_types=["personal_data"]
        )
        
        if ComplianceFramework.PDPA.value in pdpa_requirements["applicable_frameworks"]:
            logger.info("✅ PDPA compliance detected for Singapore users")
        else:
            logger.error("❌ PDPA compliance not detected for Singapore users")
            return False
        
        # Test user rights
        user_rights = gdpr_requirements.get("user_rights", [])
        required_rights = ["access", "deletion", "rectification", "portability"]
        
        if all(right in user_rights for right in required_rights):
            logger.info("✅ GDPR user rights properly identified")
        else:
            logger.error(f"❌ Missing GDPR user rights. Found: {user_rights}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Compliance frameworks test failed: {e}")
        return False


def test_data_processing_compliance():
    """Test compliant data processing."""
    logger.info("=== Testing Compliant Data Processing ===")
    
    try:
        # Test GDPR data processing
        globalization.set_region(Region.EU_WEST)
        
        test_data = {
            "user_id": "user123",
            "personal_data": "john.doe@example.com",
            "usage_metrics": {"tasks_completed": 5}
        }
        
        # Process data with consent
        result = globalization.process_data_with_compliance(
            user_id="user123",
            data=test_data,
            purpose="optimization",
            user_location="EU-DE",
            consent_given=True
        )
        
        if result and "processed_data" in result:
            logger.info("✅ Data processing with consent successful")
        else:
            logger.error("❌ Data processing with consent failed")
            return False
        
        if result["region"] in [Region.EU_WEST.value, Region.EU_CENTRAL.value]:
            logger.info("✅ Data processed in EU region (data residency)")
        else:
            logger.error(f"❌ Data not processed in EU region: {result['region']}")
            return False
        
        # Test processing without consent (should fail)
        try:
            globalization.process_data_with_compliance(
                user_id="user456",
                data=test_data,
                purpose="optimization",
                user_location="EU-DE",
                consent_given=False
            )
            logger.error("❌ Processing without consent should have failed")
            return False
        except ValueError:
            logger.info("✅ Processing correctly rejected without consent")
        
        # Test data encryption
        if "processed_data" in result:
            processed = result["processed_data"]
            if "personal_data_encrypted" in processed:
                logger.info("✅ Sensitive data encrypted")
            else:
                logger.error("❌ Sensitive data not encrypted")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data processing compliance test failed: {e}")
        return False


def test_user_data_rights():
    """Test user data rights (access, deletion, etc.)."""
    logger.info("=== Testing User Data Rights ===")
    
    try:
        # First, process some data to create records
        globalization.process_data_with_compliance(
            user_id="test_user_rights",
            data={"personal_data": "test@example.com"},
            purpose="testing",
            user_location="EU-DE",
            consent_given=True
        )
        
        # Test data access (user data report)
        report = globalization.get_user_data_report("test_user_rights")
        
        if report and "user_id" in report:
            logger.info("✅ User data report generated successfully")
        else:
            logger.error("❌ User data report generation failed")
            return False
        
        if report["total_records"] > 0:
            logger.info(f"✅ User data records found: {report['total_records']}")
        else:
            logger.error("❌ No user data records found")
            return False
        
        if "data_processing_activities" in report:
            logger.info("✅ Data processing activities included in report")
        else:
            logger.error("❌ Data processing activities not included")
            return False
        
        # Test user rights information
        if "user_rights" in report and len(report["user_rights"]) > 0:
            logger.info(f"✅ User rights information provided: {report['user_rights']}")
        else:
            logger.error("❌ User rights information missing")
            return False
        
        # Test data deletion (right to be forgotten)
        import hashlib
        verification_token = hashlib.sha256(f"delete:test_user_rights".encode()).hexdigest()[:16]
        
        deletion_result = globalization.delete_user_data("test_user_rights", verification_token)
        
        if deletion_result and "deleted_at" in deletion_result:
            logger.info("✅ User data deletion successful")
        else:
            logger.error("❌ User data deletion failed")
            return False
        
        if deletion_result["records_deleted"] > 0:
            logger.info(f"✅ Records deleted: {deletion_result['records_deleted']}")
        else:
            logger.error("❌ No records were deleted")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ User data rights test failed: {e}")
        return False


def test_globalization_decorator():
    """Test the with_globalization decorator."""
    logger.info("=== Testing Globalization Decorator ===")
    
    try:
        @with_globalization
        def test_function(agents, tasks, test_param="default"):
            """Test function with globalization support."""
            current_region = globalization.current_region
            current_language = globalization.current_language
            return {
                "region": current_region.value,
                "language": current_language.value,
                "test_param": test_param,
                "message": globalization.get_message("optimization_started")
            }
        
        # Test without globalization parameters
        agents = [Agent("test_agent", skills=["test"], capacity=1)]
        tasks = [Task("test_task", required_skills=["test"], priority=1, duration=1)]
        
        result1 = test_function(agents, tasks)
        if result1 and "region" in result1:
            logger.info(f"✅ Decorator works without parameters: {result1['region']}")
        else:
            logger.error("❌ Decorator failed without parameters")
            return False
        
        # Test with region parameter
        result2 = test_function(agents, tasks, region=Region.EU_WEST)
        if result2["region"] == Region.EU_WEST.value:
            logger.info("✅ Decorator correctly sets region")
        else:
            logger.error(f"❌ Decorator failed to set region: {result2['region']}")
            return False
        
        # Test with language parameter  
        result3 = test_function(agents, tasks, language=Language.ES)
        if result3["language"] == Language.ES.value:
            logger.info("✅ Decorator correctly sets language")
        else:
            logger.error(f"❌ Decorator failed to set language: {result3['language']}")
            return False
        
        # Test with both parameters
        result4 = test_function(
            agents, tasks, 
            region=Region.ASIA_PACIFIC, 
            language=Language.JA,
            test_param="custom"
        )
        
        if (result4["region"] == Region.ASIA_PACIFIC.value and 
            result4["language"] == Language.JA.value and
            result4["test_param"] == "custom"):
            logger.info("✅ Decorator works with multiple parameters")
        else:
            logger.error("❌ Decorator failed with multiple parameters")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Globalization decorator test failed: {e}")
        return False


def test_compliance_dashboard():
    """Test compliance dashboard and monitoring."""
    logger.info("=== Testing Compliance Dashboard ===")
    
    try:
        # Generate some test compliance data
        test_users = ["user1", "user2", "user3"]
        test_regions = [Region.US_EAST, Region.EU_WEST, Region.ASIA_PACIFIC]
        
        for i, user in enumerate(test_users):
            region = test_regions[i % len(test_regions)]
            globalization.set_region(region)
            
            # Create compliance record
            globalization.process_data_with_compliance(
                user_id=user,
                data={"usage_data": f"test_data_{i}"},
                purpose="testing",
                user_location="US" if region == Region.US_EAST else "EU",
                consent_given=True
            )
        
        # Get compliance dashboard
        dashboard = globalization.get_compliance_dashboard()
        
        if dashboard and "overview" in dashboard:
            logger.info("✅ Compliance dashboard generated")
        else:
            logger.error("❌ Compliance dashboard generation failed")
            return False
        
        overview = dashboard["overview"]
        if overview["total_compliance_records"] >= len(test_users):
            logger.info(f"✅ Compliance records tracked: {overview['total_compliance_records']}")
        else:
            logger.error("❌ Compliance records not properly tracked")
            return False
        
        if overview["active_regions"] > 0:
            logger.info(f"✅ Multi-region tracking: {overview['active_regions']} regions")
        else:
            logger.error("❌ Multi-region tracking failed")
            return False
        
        if overview["unique_users"] >= len(test_users):
            logger.info(f"✅ User tracking: {overview['unique_users']} unique users")
        else:
            logger.error("❌ User tracking failed")
            return False
        
        # Check regional distribution
        if "regional_distribution" in dashboard:
            logger.info("✅ Regional distribution metrics available")
        else:
            logger.error("❌ Regional distribution metrics missing")
            return False
        
        # Check consent metrics
        if "consent_metrics" in dashboard:
            consent_rate = dashboard["consent_metrics"]["consent_rate"]
            logger.info(f"✅ Consent metrics: {consent_rate:.2%} consent rate")
        else:
            logger.error("❌ Consent metrics missing")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Compliance dashboard test failed: {e}")
        return False


def main():
    """Run all global implementation tests."""
    logger.info("🌍 GLOBAL-FIRST IMPLEMENTATION TEST")
    logger.info("📋 Multi-region, Internationalization & Compliance")
    logger.info("=" * 80)
    
    tests = [
        ("Multi-Region Support", test_multi_region_support),
        ("Internationalization (i18n)", test_internationalization),
        ("Compliance Frameworks", test_compliance_frameworks),
        ("Data Processing Compliance", test_data_processing_compliance),
        ("User Data Rights", test_user_data_rights),
        ("Globalization Decorator", test_globalization_decorator),
        ("Compliance Dashboard", test_compliance_dashboard),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAIL - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("GLOBAL IMPLEMENTATION SUMMARY")
    logger.info(f"{'='*80}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:.<50} {status}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Global implementation complete! Ready for worldwide deployment")
        logger.info("🌐 Features implemented:")
        logger.info("  • Multi-region deployment (US, EU, Asia-Pacific)")
        logger.info("  • Internationalization (EN, ES, FR, DE, JA, ZH)")
        logger.info("  • GDPR compliance (EU users)")
        logger.info("  • CCPA compliance (California users)")
        logger.info("  • PDPA compliance (Singapore users)")
        logger.info("  • Data residency requirements")
        logger.info("  • User rights (access, deletion, rectification)")
        logger.info("  • Comprehensive compliance monitoring")
        
        # Get final compliance metrics
        dashboard = globalization.get_compliance_dashboard()
        logger.info(f"\n📊 Global Metrics:")
        logger.info(f"  • Total compliance records: {dashboard['overview']['total_compliance_records']}")
        logger.info(f"  • Active regions: {dashboard['overview']['active_regions']}")
        logger.info(f"  • Unique users: {dashboard['overview']['unique_users']}")
        logger.info(f"  • Consent rate: {dashboard['consent_metrics']['consent_rate']:.2%}")
        
        return True
    else:
        logger.error("💥 Global implementation failed! Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)