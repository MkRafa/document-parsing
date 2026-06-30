import pytest
from src.models import (
    PersonalInfo,
    InsuranceDetails,
    CoverageInfo,
    ProviderInfo,
    MedicalHistory,
    PatientInfo,
    ExtractionResult,
    Gender,
)


class TestPersonalInfo:
    """Tests for PersonalInfo model"""

    def test_personal_info_creation(self):
        """Test creating PersonalInfo object"""
        info = PersonalInfo(
            first_name="John",
            last_name="Doe",
            date_of_birth="1990-01-01",
            gender=Gender.MALE,
            email="john@example.com",
        )
        assert info.first_name == "John"
        assert info.last_name == "Doe"
        assert info.email == "john@example.com"

    def test_personal_info_optional_fields(self):
        """Test PersonalInfo with only required fields"""
        info = PersonalInfo(first_name="Jane", last_name="Smith")
        assert info.first_name == "Jane"
        assert info.email is None
        assert info.phone is None


class TestInsuranceDetails:
    """Tests for InsuranceDetails model"""

    def test_insurance_details_creation(self):
        """Test creating InsuranceDetails"""
        details = InsuranceDetails(
            member_id="123456789",
            policy_number="POL-001",
            plan_type="PPO",
        )
        assert details.member_id == "123456789"
        assert details.policy_number == "POL-001"

    def test_insurance_details_required_field(self):
        """Test InsuranceDetails member_id is required"""
        details = InsuranceDetails(member_id="123456789")
        assert details.member_id == "123456789"


class TestCoverageInfo:
    """Tests for CoverageInfo model"""

    def test_coverage_info_creation(self):
        """Test creating CoverageInfo"""
        coverage = CoverageInfo(
            deductible=500.0,
            out_of_pocket_max=5000.0,
            copay_visit=25.0,
            covered_services=["primary_care", "specialists"],
        )
        assert coverage.deductible == 500.0
        assert len(coverage.covered_services) == 2

    def test_coverage_info_empty(self):
        """Test creating empty CoverageInfo"""
        coverage = CoverageInfo()
        assert coverage.deductible is None
        assert coverage.covered_services == []


class TestPatientInfo:
    """Tests for complete PatientInfo model"""

    def test_patient_info_creation(self):
        """Test creating complete PatientInfo"""
        patient = PatientInfo(
            personal_info=PersonalInfo(first_name="John", last_name="Doe"),
            insurance_details=InsuranceDetails(member_id="123456789"),
            coverage_info=CoverageInfo(deductible=500.0),
            provider_info=ProviderInfo(),
            medical_history=MedicalHistory(),
        )
        assert patient.personal_info.first_name == "John"
        assert patient.insurance_details.member_id == "123456789"

    def test_patient_info_confidence(self):
        """Test confidence field"""
        patient = PatientInfo(
            personal_info=PersonalInfo(first_name="John", last_name="Doe"),
            insurance_details=InsuranceDetails(member_id="123456789"),
            coverage_info=CoverageInfo(),
            provider_info=ProviderInfo(),
            medical_history=MedicalHistory(),
            extraction_confidence=0.95,
        )
        assert patient.extraction_confidence == 0.95
        assert 0.0 <= patient.extraction_confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
