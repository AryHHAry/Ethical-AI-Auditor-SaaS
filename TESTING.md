# 🧪 Testing Guide - Ethical AI Auditor SaaS

Comprehensive testing procedures to ensure the application works correctly.

## Table of Contents
1. [Quick Start Testing](#quick-start)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [User Acceptance Testing](#uat)
5. [Performance Testing](#performance)
6. [Security Testing](#security)
7. [Blockchain Testing](#blockchain)

---

## Quick Start Testing {#quick-start}

### 1. Installation Verification

```bash
# Verify Python version
python --version  # Should be 3.8+

# Verify dependencies
pip list | grep streamlit
pip list | grep torch
pip list | grep web3

# Run application
streamlit run app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

### 2. Basic Functionality Test

**Test Case 1: Sample Data Generation**
- Navigate to "Upload & Configure"
- Select "Use Sample Data"
- Click "Generate Sample Dataset"
- ✅ Expected: Dataset appears with 1000 rows, 5 columns
- ✅ Expected: No errors in console

**Test Case 2: Simple Audit**
- Click "Save Configuration" (default settings)
- Navigate to "Run Audit"
- Click "Start Audit"
- ✅ Expected: Progress bar completes
- ✅ Expected: Success message appears
- ✅ Expected: Audit completes in < 10 seconds

**Test Case 3: View Results**
- Navigate to "View Results"
- ✅ Expected: Fairness score displayed (0-100)
- ✅ Expected: Charts render without errors
- ✅ Expected: Report downloadable

---

## Unit Testing {#unit-testing}

### Test Bias Detection Functions

Create `test_bias_detection.py`:

```python
import pytest
import pandas as pd
import numpy as np
from app import (
    calculate_demographic_parity,
    calculate_disparate_impact,
    calculate_equalized_odds,
    generate_sample_data
)

def test_demographic_parity_perfect_equality():
    """Test demographic parity with perfectly equal outcomes"""
    df = pd.DataFrame({
        'gender': ['M', 'M', 'F', 'F'],
        'approved': [1, 1, 1, 1]
    })
    
    result = calculate_demographic_parity(df, 'gender', 'approved')
    
    assert result['difference'] == 0.0
    assert result['passed'] == True

def test_demographic_parity_with_bias():
    """Test demographic parity with biased outcomes"""
    df = pd.DataFrame({
        'gender': ['M'] * 50 + ['F'] * 50,
        'approved': [1] * 40 + [0] * 10 + [1] * 20 + [0] * 30
    })
    
    result = calculate_demographic_parity(df, 'gender', 'approved')
    
    # M: 40/50 = 0.8, F: 20/50 = 0.4, diff = 0.4
    assert result['difference'] > 0.3
    assert result['passed'] == False

def test_disparate_impact_80_percent_rule():
    """Test disparate impact 80% rule"""
    df = pd.DataFrame({
        'race': ['A'] * 100 + ['B'] * 100,
        'approved': [1] * 90 + [0] * 10 + [1] * 70 + [0] * 30
    })
    
    result = calculate_disparate_impact(df, 'race', 'approved')
    
    # A: 90/100 = 0.9, B: 70/100 = 0.7, ratio = 0.7/0.9 ≈ 0.78
    assert result['ratio'] < 0.8
    assert result['passed'] == False

def test_sample_data_generation():
    """Test sample data generation"""
    df = generate_sample_data()
    
    assert len(df) == 1000
    assert 'gender' in df.columns
    assert 'race' in df.columns
    assert 'approved' in df.columns
    assert df['approved'].isin([0, 1]).all()

def test_equalized_odds():
    """Test equalized odds calculation"""
    df = pd.DataFrame({
        'gender': ['M'] * 100 + ['F'] * 100,
        'approved': [1] * 60 + [0] * 40 + [1] * 50 + [0] * 50,
        'predicted': [1] * 55 + [0] * 45 + [1] * 48 + [0] * 52
    })
    
    result = calculate_equalized_odds(df, 'gender', 'approved', 'predicted')
    
    assert 'tpr' in result
    assert 'fpr' in result
    assert 'max_difference' in result

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

Run tests:
```bash
pip install pytest
pytest test_bias_detection.py -v
```

### Test Privacy Functions

Create `test_privacy.py`:

```python
import pytest
import pandas as pd
from app import detect_pii_columns, calculate_privacy_score

def test_pii_detection_with_pii():
    """Test PII detection with obvious PII columns"""
    df = pd.DataFrame({
        'name': ['John Doe', 'Jane Smith'],
        'email': ['john@example.com', 'jane@example.com'],
        'age': [30, 25],
        'score': [85, 90]
    })
    
    pii_cols = detect_pii_columns(df)
    
    assert 'name' in pii_cols
    assert 'email' in pii_cols
    assert 'age' not in pii_cols
    assert 'score' not in pii_cols

def test_pii_detection_no_pii():
    """Test PII detection with no PII columns"""
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    
    pii_cols = detect_pii_columns(df)
    
    assert len(pii_cols) == 0

def test_privacy_score_high_risk():
    """Test privacy score with high-risk data"""
    df = pd.DataFrame({
        'ssn': ['123-45-6789', '987-65-4321'],
        'name': ['John', 'Jane'],
        'unique_id': [1, 2]
    })
    
    result = calculate_privacy_score(df)
    
    assert result['score'] < 50
    assert result['risk_level'] == 'High'

def test_privacy_score_low_risk():
    """Test privacy score with low-risk data"""
    df = pd.DataFrame({
        'feature1': range(100),
        'feature2': range(100),
        'category': ['A', 'B'] * 50
    })
    
    result = calculate_privacy_score(df)
    
    assert result['score'] > 50

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Test Model Analysis

Create `test_model.py`:

```python
import pytest
import torch
import torch.nn as nn
from app import analyze_model_structure, generate_sample_model

def test_model_structure_analysis():
    """Test model structure analysis"""
    model = generate_sample_model()
    
    result = analyze_model_structure(model)
    
    assert result is not None
    assert 'total_layers' in result
    assert 'total_parameters' in result
    assert result['total_layers'] > 0
    assert result['total_parameters'] > 0

def test_sample_model_generation():
    """Test sample model generation"""
    model = generate_sample_model()
    
    assert isinstance(model, nn.Module)
    
    # Test forward pass
    x = torch.randn(10, 2)
    output = model(x)
    
    assert output.shape == (10, 1)
    assert (output >= 0).all() and (output <= 1).all()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Integration Testing {#integration-testing}

### End-to-End Workflow Tests

Create `test_integration.py`:

```python
import pytest
import pandas as pd
from app import (
    generate_sample_data,
    run_bias_audit,
    calculate_fairness_score,
    calculate_privacy_score,
    generate_audit_report
)

def test_complete_audit_workflow():
    """Test complete audit workflow from data to report"""
    # 1. Generate data
    df = generate_sample_data()
    assert len(df) > 0
    
    # 2. Run bias audit
    bias_results = run_bias_audit(
        df, 
        protected_attrs=['gender', 'race'],
        outcome_col='approved'
    )
    assert 'gender' in bias_results
    assert 'race' in bias_results
    
    # 3. Calculate fairness score
    fairness_score = calculate_fairness_score(bias_results)
    assert 0 <= fairness_score['score'] <= 100
    assert 'grade' in fairness_score
    
    # 4. Calculate privacy score
    privacy_score = calculate_privacy_score(df)
    assert 0 <= privacy_score['score'] <= 100
    
    # 5. Generate report
    audit_results = {
        'audit_id': 'test123',
        'bias_results': bias_results,
        'fairness_score': fairness_score,
        'privacy_score': privacy_score
    }
    
    report = generate_audit_report(audit_results)
    assert len(report) > 0
    assert 'ETHICAL AI AUDIT REPORT' in report
    assert 'FAIRNESS SCORE' in report.upper()

def test_csv_upload_workflow():
    """Test CSV upload and processing workflow"""
    # Create test CSV
    df = generate_sample_data()
    csv_path = '/tmp/test_data.csv'
    df.to_csv(csv_path, index=False)
    
    # Load CSV
    loaded_df = pd.read_csv(csv_path)
    
    assert len(loaded_df) == len(df)
    assert list(loaded_df.columns) == list(df.columns)
    
    # Run audit
    bias_results = run_bias_audit(
        loaded_df,
        protected_attrs=['gender'],
        outcome_col='approved'
    )
    
    assert bias_results is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## User Acceptance Testing (UAT) {#uat}

### Manual Test Cases

#### TC-001: Upload CSV File

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to "Upload & Configure" | Page loads successfully |
| 2 | Click "Upload your dataset (CSV format)" | File picker opens |
| 3 | Select valid CSV file (<10MB) | File uploads successfully |
| 4 | - | Preview shows first 10 rows |
| 5 | - | Column types displayed correctly |

**Pass Criteria**: File uploads and displays without errors

#### TC-002: Configure Audit Settings

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Select outcome column from dropdown | Column selected |
| 2 | Select protected attributes | Multiple selection works |
| 3 | Check audit options (bias, fairness, privacy) | Checkboxes toggle |
| 4 | Click "Save Configuration" | Success message appears |

**Pass Criteria**: Configuration saved successfully

#### TC-003: Run Complete Audit

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to "Run Audit" | Configuration displayed |
| 2 | Click "Start Audit" | Progress bar appears |
| 3 | Wait for completion | Progress reaches 100% |
| 4 | - | Success message shown |
| 5 | - | Audit completes in <10s for sample data |

**Pass Criteria**: Audit completes successfully

#### TC-004: View and Download Results

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to "View Results" | Results dashboard loads |
| 2 | Check metrics display | Fairness and privacy scores shown |
| 3 | Expand chart sections | All charts render correctly |
| 4 | Click "Download TXT Report" | Report downloads |
| 5 | Open downloaded report | Report contains all sections |

**Pass Criteria**: Results viewable and downloadable

#### TC-005: Blockchain Integration

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to "Blockchain Verification" | Page loads |
| 2 | Click "Log Audit Record" | Transaction initiated |
| 3 | Wait for confirmation | Success message appears |
| 4 | Enter Ethereum address | Address accepted |
| 5 | Click "Mint NFT Certificate" | NFT minting successful |
| 6 | - | Certificate preview displayed |

**Pass Criteria**: Blockchain operations complete (or simulate correctly)

### Edge Cases and Error Handling

#### TC-006: File Size Limit

| Input | Expected Behavior |
|-------|-------------------|
| File > 10MB | Error: "File too large! Max size: 10MB" |
| File = 0 bytes | Error: "Invalid file" |
| Non-CSV file with .csv extension | Error: "Error loading file" |

#### TC-007: Invalid Data

| Input | Expected Behavior |
|-------|-------------------|
| CSV with no outcome column | Warning shown |
| CSV with all missing values | Handled gracefully |
| CSV with non-numeric outcome | Error or auto-conversion |
| Empty CSV | Error: "No data found" |

#### TC-008: Configuration Errors

| Scenario | Expected Behavior |
|----------|-------------------|
| No protected attributes selected | Warning or default selection |
| No audit checks selected | Warning: "Select at least one check" |
| Invalid column names | Error with helpful message |

---

## Performance Testing {#performance}

### Load Testing

Create `test_performance.py`:

```python
import time
import pandas as pd
import numpy as np
from app import run_bias_audit, generate_sample_data

def test_small_dataset_performance():
    """Test performance with small dataset (100 rows)"""
    df = pd.DataFrame({
        'gender': np.random.choice(['M', 'F'], 100),
        'approved': np.random.choice([0, 1], 100)
    })
    
    start = time.time()
    result = run_bias_audit(df, ['gender'], 'approved')
    duration = time.time() - start
    
    assert duration < 2.0  # Should complete in < 2 seconds
    print(f"Small dataset: {duration:.2f}s")

def test_medium_dataset_performance():
    """Test performance with medium dataset (1000 rows)"""
    df = generate_sample_data()
    
    start = time.time()
    result = run_bias_audit(df, ['gender', 'race'], 'approved')
    duration = time.time() - start
    
    assert duration < 10.0  # Should complete in < 10 seconds
    print(f"Medium dataset: {duration:.2f}s")

def test_large_dataset_performance():
    """Test performance with large dataset (10000 rows)"""
    np.random.seed(42)
    df = pd.DataFrame({
        'gender': np.random.choice(['M', 'F'], 10000),
        'race': np.random.choice(['A', 'B', 'C', 'D'], 10000),
        'approved': np.random.choice([0, 1], 10000)
    })
    
    start = time.time()
    result = run_bias_audit(df, ['gender'], 'approved')
    duration = time.time() - start
    
    assert duration < 30.0  # Should complete in < 30 seconds
    print(f"Large dataset: {duration:.2f}s")

if __name__ == '__main__':
    test_small_dataset_performance()
    test_medium_dataset_performance()
    test_large_dataset_performance()
```

### Benchmark Results (Expected)

| Dataset Size | Rows | Columns | Protected Attrs | Time (s) | Status |
|--------------|------|---------|-----------------|----------|--------|
| Small | 100 | 5 | 1 | <2 | ✅ Pass |
| Medium | 1,000 | 5 | 2 | <10 | ✅ Pass |
| Large | 10,000 | 5 | 2 | <30 | ✅ Pass |
| XLarge | 100,000 | 10 | 3 | <60 | ⚠️ Warning |

---

## Security Testing {#security}

### Security Test Cases

#### SEC-001: File Upload Security

```python
def test_file_size_validation():
    """Test file size limit enforcement"""
    # Attempt to upload file > 10MB
    # Expected: Rejection with error message

def test_file_type_validation():
    """Test file type restrictions"""
    # Attempt to upload .exe, .sh, .py files
    # Expected: Only .csv, .pth, .pkl accepted

def test_malicious_csv():
    """Test CSV with malicious content"""
    # CSV with formula injection: =cmd|'/c notepad'
    # Expected: Content sanitized or rejected
```

#### SEC-002: Input Sanitization

```python
def test_sql_injection_prevention():
    """Test SQL injection attempts"""
    # If database features added
    # Input: "'; DROP TABLE users; --"
    # Expected: Escaped/rejected

def test_xss_prevention():
    """Test XSS attack prevention"""
    # Input: "<script>alert('xss')</script>"
    # Expected: HTML encoded or rejected
```

#### SEC-003: Secrets Management

**Checklist:**
- [ ] No hardcoded API keys in code
- [ ] secrets.toml in .gitignore
- [ ] Private keys never logged
- [ ] Environment variables used for production
- [ ] Secrets rotation procedure documented

---

## Blockchain Testing {#blockchain}

### Blockchain Test Cases

#### BC-001: Connection Testing

```python
def test_infura_connection():
    """Test connection to Infura"""
    from web3 import Web3
    
    infura_url = "https://sepolia.infura.io/v3/YOUR_KEY"
    web3 = Web3(Web3.HTTPProvider(infura_url))
    
    assert web3.is_connected()
    assert web3.eth.chain_id == 11155111  # Sepolia

def test_wallet_balance():
    """Test wallet has sufficient balance"""
    # Check Sepolia testnet ETH balance
    # Expected: > 0.01 ETH for testing
```

#### BC-002: Transaction Testing

```python
def test_audit_logging():
    """Test audit log transaction"""
    from app import BlockchainIntegration
    
    bc = BlockchainIntegration()
    result = bc.log_audit("test_hash_123", "Test Audit")
    
    assert result['success'] == True
    assert 'tx_hash' in result
    assert len(result['tx_hash']) == 66  # 0x + 64 chars

def test_nft_minting():
    """Test NFT certificate minting"""
    from app import BlockchainIntegration
    
    bc = BlockchainIntegration()
    result = bc.mint_nft_certificate(
        "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb4",
        "audit_hash_456"
    )
    
    assert result['success'] == True
    assert 'token_id' in result
```

#### BC-003: Simulation Mode Testing

```python
def test_simulation_mode():
    """Test blockchain simulation when no config"""
    from app import BlockchainIntegration
    
    bc = BlockchainIntegration()
    
    if bc.simulated:
        result = bc.log_audit("hash", "type")
        assert result.get('simulated') == True
        assert 'tx_hash' in result
```

---

## Regression Testing

### Before Each Release

**Checklist:**
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Sample data generation works
- [ ] CSV upload works for valid files
- [ ] Model upload works (.pth and .pkl)
- [ ] All audit types complete successfully
- [ ] Visualizations render correctly
- [ ] Reports downloadable and well-formatted
- [ ] Blockchain features work (real or simulated)
- [ ] No console errors
- [ ] Performance acceptable (<10s for typical audit)
- [ ] Mobile responsive (if applicable)
- [ ] Accessibility standards met (WCAG 2.1)

---

## Test Data Sets

### Test Data 1: Balanced Dataset (No Bias)
```csv
gender,race,income,approved
M,White,50000,1
F,White,51000,1
M,Black,49000,1
F,Black,50500,1
M,Asian,50200,1
F,Asian,49800,1
```

### Test Data 2: Biased Dataset
```csv
gender,race,income,approved
M,White,50000,1
M,White,51000,1
M,White,49000,1
F,Black,60000,0
F,Black,61000,0
F,Asian,59000,0
```

### Test Data 3: Large Scale Test
Generate programmatically:
```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

df = pd.DataFrame({
    'gender': np.random.choice(['M', 'F'], n),
    'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n),
    'age': np.random.randint(18, 80, n),
    'income': np.random.randint(20000, 150000, n),
    'credit_score': np.random.randint(300, 850, n),
    'approved': np.random.choice([0, 1], n)
})

df.to_csv('large_test_data.csv', index=False)
```

---

## Continuous Integration Setup

### GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Test Ethical AI Auditor

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest test_*.py -v --cov=app
    
    - name: Test app startup
      run: |
        timeout 30 streamlit run app.py --server.headless true &
        sleep 20
        curl http://localhost:8501
```

---

## Bug Report Template

When reporting bugs, include:

```markdown
## Bug Description
[Clear description of the issue]

## Steps to Reproduce
1. Navigate to...
2. Click on...
3. Upload file...
4. See error

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happened]

## Screenshots
[If applicable]

## Environment
- OS: [e.g., Windows 10, macOS 12, Ubuntu 20.04]
- Browser: [e.g., Chrome 96, Firefox 95]
- Python Version: [e.g., 3.9.7]
- App Version: [e.g., 1.0.0]

## Additional Context
[Any other relevant information]
```

---

**Happy Testing! 🧪**
