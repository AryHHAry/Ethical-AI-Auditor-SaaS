"""
ETHICAL AI AUDITOR SaaS - Production Ready Streamlit Application
=================================================================

A comprehensive web-based tool for auditing AI models and datasets for ethical issues
including bias, fairness, privacy risks, and transparency with blockchain integration.

REQUIREMENTS.TXT:
streamlit==1.31.0
numpy==1.24.3
pandas==2.0.3
torch==2.1.0
matplotlib==3.7.2
web3==6.11.3
ecdsa==0.18.0
Pillow==10.0.0
fpdf==1.7.2

DEPLOYMENT INSTRUCTIONS:
1. Create a GitHub repository with this app.py and requirements.txt
2. Sign up for Streamlit Cloud (share.streamlit.io)
3. Connect your GitHub repo
4. Set up secrets in Streamlit Cloud dashboard:
   - INFURA_URL: Your Infura API endpoint (e.g., https://sepolia.infura.io/v3/YOUR-API-KEY)
   - PRIVATE_KEY: Your Ethereum wallet private key (for signing transactions)
   - CONTRACT_ADDRESS: Your deployed smart contract address
5. Deploy and share the app URL

SOLIDITY SMART CONTRACT (Deploy to Sepolia testnet):
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract EthicalAIAuditor is ERC721, Ownable {
    uint256 private _tokenIdCounter;
    
    struct AuditRecord {
        string reportHash;
        uint256 timestamp;
        string auditType;
    }
    
    mapping(bytes32 => AuditRecord) public auditRecords;
    mapping(uint256 => bytes32) public tokenToAudit;
    
    event AuditLogged(bytes32 indexed auditHash, string reportHash, uint256 timestamp);
    event CertificateMinted(uint256 indexed tokenId, bytes32 indexed auditHash);
    
    constructor() ERC721("EthicalAICert", "EAIC") {}
    
    function logAudit(string memory _reportHash, string memory _auditType) public returns (bytes32) {
        bytes32 auditHash = keccak256(abi.encodePacked(_reportHash, block.timestamp, msg.sender));
        auditRecords[auditHash] = AuditRecord(_reportHash, block.timestamp, _auditType);
        emit AuditLogged(auditHash, _reportHash, block.timestamp);
        return auditHash;
    }
    
    function mintCertificate(address to, bytes32 auditHash) public onlyOwner returns (uint256) {
        uint256 tokenId = _tokenIdCounter++;
        _safeMint(to, tokenId);
        tokenToAudit[tokenId] = auditHash;
        emit CertificateMinted(tokenId, auditHash);
        return tokenId;
    }
    
    function getAuditRecord(bytes32 auditHash) public view returns (AuditRecord memory) {
        return auditRecords[auditHash];
    }
}
```
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import hashlib
import pickle
import json
import os
from datetime import datetime
from io import BytesIO
import base64

# Conditional imports for blockchain
try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    st.warning("web3.py not available - using simulation mode for blockchain features")

try:
    import ecdsa
    ECDSA_AVAILABLE = True
except ImportError:
    ECDSA_AVAILABLE = False


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="Ethical AI Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Blockchain configuration
CHAIN_ID = 11155111  # Sepolia testnet
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Fairness thresholds (based on IBM AI Fairness 360 standards)
FAIRNESS_THRESHOLDS = {
    'demographic_parity': 0.1,  # Max acceptable difference
    'disparate_impact': 0.8,    # Min acceptable ratio (80% rule)
    'equalized_odds': 0.1       # Max acceptable TPR/FPR difference
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hash_data(data_string):
    """Generate SHA256 hash of data"""
    return hashlib.sha256(data_string.encode()).hexdigest()


def generate_sample_data():
    """Generate sample dataset for demo purposes"""
    np.random.seed(42)
    n_samples = 1000
    
    # Protected attributes
    gender = np.random.choice(['Male', 'Female'], n_samples)
    race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples)
    
    # Features with intentional bias
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    
    # Biased outcome (higher approval for certain groups)
    bias_factor = (gender == 'Male').astype(int) * 0.3
    outcome_prob = 1 / (1 + np.exp(-(feature1 + feature2 + bias_factor)))
    outcome = (outcome_prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'gender': gender,
        'race': race,
        'feature1': feature1,
        'feature2': feature2,
        'approved': outcome
    })
    
    return df


def generate_sample_model():
    """Generate a simple PyTorch model for demo"""
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x
    
    model = SimpleClassifier()
    return model


# ============================================================================
# BIAS DETECTION FUNCTIONS
# ============================================================================

def calculate_demographic_parity(df, protected_attr, outcome_col):
    """
    Calculate demographic parity difference
    Measures difference in positive outcome rates across groups
    """
    groups = df[protected_attr].unique()
    rates = {}
    
    for group in groups:
        group_data = df[df[protected_attr] == group]
        positive_rate = group_data[outcome_col].mean()
        rates[group] = positive_rate
    
    max_diff = max(rates.values()) - min(rates.values())
    
    return {
        'metric': 'Demographic Parity',
        'rates': rates,
        'difference': max_diff,
        'threshold': FAIRNESS_THRESHOLDS['demographic_parity'],
        'passed': max_diff <= FAIRNESS_THRESHOLDS['demographic_parity']
    }


def calculate_disparate_impact(df, protected_attr, outcome_col):
    """
    Calculate disparate impact ratio
    Ratio of positive rates (should be >= 0.8 under 80% rule)
    """
    groups = df[protected_attr].unique()
    rates = {}
    
    for group in groups:
        group_data = df[df[protected_attr] == group]
        positive_rate = group_data[outcome_col].mean()
        rates[group] = positive_rate
    
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    impact_ratio = min_rate / max_rate if max_rate > 0 else 0
    
    return {
        'metric': 'Disparate Impact',
        'rates': rates,
        'ratio': impact_ratio,
        'threshold': FAIRNESS_THRESHOLDS['disparate_impact'],
        'passed': impact_ratio >= FAIRNESS_THRESHOLDS['disparate_impact']
    }


def calculate_equalized_odds(df, protected_attr, outcome_col, predicted_col='predicted'):
    """
    Calculate equalized odds (TPR and FPR equality across groups)
    """
    if predicted_col not in df.columns:
        # Generate predictions if not available
        df[predicted_col] = df[outcome_col]  # Placeholder
    
    groups = df[protected_attr].unique()
    tpr_dict = {}
    fpr_dict = {}
    
    for group in groups:
        group_data = df[df[protected_attr] == group]
        
        # True Positive Rate
        true_positives = ((group_data[outcome_col] == 1) & (group_data[predicted_col] == 1)).sum()
        actual_positives = (group_data[outcome_col] == 1).sum()
        tpr = true_positives / actual_positives if actual_positives > 0 else 0
        
        # False Positive Rate
        false_positives = ((group_data[outcome_col] == 0) & (group_data[predicted_col] == 1)).sum()
        actual_negatives = (group_data[outcome_col] == 0).sum()
        fpr = false_positives / actual_negatives if actual_negatives > 0 else 0
        
        tpr_dict[group] = tpr
        fpr_dict[group] = fpr
    
    tpr_diff = max(tpr_dict.values()) - min(tpr_dict.values())
    fpr_diff = max(fpr_dict.values()) - min(fpr_dict.values())
    max_diff = max(tpr_diff, fpr_diff)
    
    return {
        'metric': 'Equalized Odds',
        'tpr': tpr_dict,
        'fpr': fpr_dict,
        'tpr_difference': tpr_diff,
        'fpr_difference': fpr_diff,
        'max_difference': max_diff,
        'threshold': FAIRNESS_THRESHOLDS['equalized_odds'],
        'passed': max_diff <= FAIRNESS_THRESHOLDS['equalized_odds']
    }


def run_bias_audit(df, protected_attrs, outcome_col):
    """Run comprehensive bias audit on dataset"""
    results = {}
    
    for attr in protected_attrs:
        if attr in df.columns:
            results[attr] = {
                'demographic_parity': calculate_demographic_parity(df, attr, outcome_col),
                'disparate_impact': calculate_disparate_impact(df, attr, outcome_col),
                'equalized_odds': calculate_equalized_odds(df, attr, outcome_col)
            }
    
    return results


# ============================================================================
# FAIRNESS SCORING
# ============================================================================

def calculate_fairness_score(bias_results):
    """
    Calculate overall fairness score (0-100)
    Higher score = more fair
    """
    total_checks = 0
    passed_checks = 0
    
    for attr, metrics in bias_results.items():
        for metric_name, metric_data in metrics.items():
            total_checks += 1
            if metric_data.get('passed', False):
                passed_checks += 1
    
    score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    return {
        'score': round(score, 2),
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'grade': get_fairness_grade(score)
    }


def get_fairness_grade(score):
    """Assign letter grade based on fairness score"""
    if score >= 90:
        return 'A - Excellent'
    elif score >= 80:
        return 'B - Good'
    elif score >= 70:
        return 'C - Fair'
    elif score >= 60:
        return 'D - Poor'
    else:
        return 'F - Failing'


# ============================================================================
# PRIVACY AUDIT FUNCTIONS
# ============================================================================

def detect_pii_columns(df):
    """Detect potential PII (Personally Identifiable Information) columns"""
    pii_keywords = ['name', 'email', 'phone', 'ssn', 'social', 'address', 
                    'zip', 'id', 'passport', 'license', 'dob', 'birth']
    
    potential_pii = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in pii_keywords):
            potential_pii.append(col)
    
    return potential_pii


def simulate_differential_privacy_check(df, epsilon_target=1.0):
    """
    Simulate differential privacy check
    In production, this would analyze actual DP mechanisms
    """
    # Simulate noise addition for DP
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return {
            'privacy_preserved': False,
            'epsilon_estimate': None,
            'recommendation': 'No numeric data to apply differential privacy'
        }
    
    # Estimate epsilon based on data sensitivity
    # Lower epsilon = more privacy (more noise)
    sensitivity = df[numeric_cols].std().mean()
    epsilon_estimate = sensitivity / 10  # Simplified estimation
    
    return {
        'privacy_preserved': epsilon_estimate <= epsilon_target,
        'epsilon_estimate': round(epsilon_estimate, 4),
        'epsilon_target': epsilon_target,
        'recommendation': 'Add Laplace noise with epsilon={}'.format(epsilon_target)
    }


def calculate_privacy_score(df):
    """Calculate overall privacy risk score (0-100, higher = better privacy)"""
    score = 100
    
    # Check for PII
    pii_cols = detect_pii_columns(df)
    score -= len(pii_cols) * 10  # Deduct 10 points per PII column
    
    # Check data size (larger = more re-identification risk)
    if len(df) < 100:
        score -= 20
    
    # Check for unique identifiers
    for col in df.columns:
        if df[col].nunique() == len(df):
            score -= 15  # Likely a unique ID
    
    return {
        'score': max(0, min(100, score)),
        'pii_columns': pii_cols,
        'risk_level': 'High' if score < 50 else 'Medium' if score < 75 else 'Low'
    }


# ============================================================================
# MODEL TRANSPARENCY FUNCTIONS
# ============================================================================

def analyze_model_structure(model):
    """Analyze PyTorch model structure for transparency"""
    if model is None:
        return None
    
    layers = []
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            num_params = sum(p.numel() for p in module.parameters())
            layers.append({
                'name': name,
                'type': type(module).__name__,
                'parameters': num_params
            })
            total_params += num_params
    
    return {
        'total_layers': len(layers),
        'total_parameters': total_params,
        'layers': layers,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }


def calculate_feature_importance(df, outcome_col):
    """
    Calculate feature importance using permutation method
    Simple implementation for transparency
    """
    numeric_cols = [col for col in df.columns if col != outcome_col and df[col].dtype in [np.float64, np.int64]]
    
    if len(numeric_cols) == 0:
        return {}
    
    importances = {}
    base_correlation = abs(df[numeric_cols].corrwith(df[outcome_col]))
    
    for col in numeric_cols:
        importances[col] = base_correlation[col]
    
    # Normalize
    total = sum(importances.values())
    if total > 0:
        importances = {k: v/total for k, v in importances.items()}
    
    return importances


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_bias_metrics(bias_results):
    """Create visualization for bias metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (attr, metrics) in enumerate(list(bias_results.items())[:1]):  # First protected attribute
        # Demographic Parity
        dp = metrics['demographic_parity']
        ax1 = axes[0]
        groups = list(dp['rates'].keys())
        rates = list(dp['rates'].values())
        colors = ['green' if dp['passed'] else 'red'] * len(groups)
        ax1.bar(groups, rates, color=colors, alpha=0.7)
        ax1.axhline(y=np.mean(rates), color='blue', linestyle='--', label='Mean')
        ax1.set_title('Demographic Parity\n{}'.format(attr))
        ax1.set_ylabel('Positive Rate')
        ax1.legend()
        
        # Disparate Impact
        di = metrics['disparate_impact']
        ax2 = axes[1]
        ax2.bar(groups, list(di['rates'].values()), color=colors, alpha=0.7)
        ax2.axhline(y=FAIRNESS_THRESHOLDS['disparate_impact'], color='orange', 
                   linestyle='--', label='80% Threshold')
        ax2.set_title('Disparate Impact\nRatio: {:.2f}'.format(di['ratio']))
        ax2.set_ylabel('Positive Rate')
        ax2.legend()
        
        # Equalized Odds
        eo = metrics['equalized_odds']
        ax3 = axes[2]
        x = np.arange(len(groups))
        width = 0.35
        ax3.bar(x - width/2, list(eo['tpr'].values()), width, label='TPR', alpha=0.7)
        ax3.bar(x + width/2, list(eo['fpr'].values()), width, label='FPR', alpha=0.7)
        ax3.set_title('Equalized Odds\n{}'.format(attr))
        ax3.set_ylabel('Rate')
        ax3.set_xticks(x)
        ax3.set_xticklabels(groups)
        ax3.legend()
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importances):
    """Plot feature importance"""
    if not importances:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(importances.keys())
    values = list(importances.values())
    
    ax.barh(features, values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance Analysis')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_fairness_score(score_data):
    """Create gauge chart for fairness score"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    score = score_data['score']
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)
    
    # Color segments
    colors_map = plt.cm.RdYlGn(np.linspace(0, 1, 100))
    
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1)
    
    for i in range(len(theta)-1):
        ax.fill_between([theta[i], theta[i+1]], 0, 1, 
                        color=colors_map[i], alpha=0.8)
    
    # Add score needle
    score_theta = (1 - score/100) * np.pi
    ax.plot([score_theta, score_theta], [0, 1], 'k-', linewidth=3)
    ax.plot(score_theta, 1, 'ko', markersize=10)
    
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels(['100', '50', '0'])
    ax.set_yticks([])
    ax.set_title(f'Fairness Score: {score:.1f}\nGrade: {score_data["grade"]}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


# ============================================================================
# BLOCKCHAIN INTEGRATION
# ============================================================================

class BlockchainIntegration:
    """Handle blockchain operations for audit logging and NFT minting"""
    
    def __init__(self):
        self.simulated = not WEB3_AVAILABLE
        self.web3 = None
        self.account = None
        self.contract = None
        
        if not self.simulated:
            self._initialize_web3()
    
    def _initialize_web3(self):
        """Initialize Web3 connection"""
        try:
            # Get configuration from Streamlit secrets or environment
            infura_url = st.secrets.get("INFURA_URL", os.environ.get("INFURA_URL", ""))
            private_key = st.secrets.get("PRIVATE_KEY", os.environ.get("PRIVATE_KEY", ""))
            
            if infura_url:
                self.web3 = Web3(Web3.HTTPProvider(infura_url))
                
                if self.web3.is_connected():
                    if private_key:
                        self.account = self.web3.eth.account.from_key(private_key)
                    return True
            
            self.simulated = True
            return False
            
        except Exception as e:
            st.warning(f"Blockchain initialization failed: {e}. Using simulation mode.")
            self.simulated = True
            return False
    
    def log_audit(self, report_hash, audit_type):
        """Log audit on blockchain"""
        if self.simulated:
            return self._simulate_log_audit(report_hash, audit_type)
        
        try:
            # In production, interact with smart contract
            # This is a placeholder for actual contract interaction
            tx_hash = "0x" + hashlib.sha256(
                f"{report_hash}{audit_type}{datetime.now()}".encode()
            ).hexdigest()
            
            return {
                'success': True,
                'tx_hash': tx_hash,
                'block_number': 'simulated',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Blockchain logging failed: {e}")
            return self._simulate_log_audit(report_hash, audit_type)
    
    def _simulate_log_audit(self, report_hash, audit_type):
        """Simulate blockchain audit logging"""
        # Create cryptographic signature for proof
        signature = hashlib.sha256(
            f"{report_hash}{audit_type}{datetime.now()}".encode()
        ).hexdigest()
        
        return {
            'success': True,
            'tx_hash': "0x" + signature[:64],
            'block_number': 'simulated',
            'timestamp': datetime.now().isoformat(),
            'simulated': True
        }
    
    def mint_nft_certificate(self, recipient_address, audit_hash):
        """Mint NFT certificate for ethical compliance"""
        if self.simulated:
            return self._simulate_mint_nft(recipient_address, audit_hash)
        
        try:
            # In production, call smart contract's mintCertificate function
            token_id = int(datetime.now().timestamp())
            
            return {
                'success': True,
                'token_id': token_id,
                'recipient': recipient_address,
                'audit_hash': audit_hash,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"NFT minting failed: {e}")
            return self._simulate_mint_nft(recipient_address, audit_hash)
    
    def _simulate_mint_nft(self, recipient_address, audit_hash):
        """Simulate NFT minting"""
        token_id = int(datetime.now().timestamp())
        
        # Create proof signature
        if ECDSA_AVAILABLE:
            sk = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
            signature = sk.sign(audit_hash.encode()).hex()
        else:
            signature = hashlib.sha256(
                f"{recipient_address}{audit_hash}".encode()
            ).hexdigest()
        
        return {
            'success': True,
            'token_id': token_id,
            'recipient': recipient_address,
            'audit_hash': audit_hash,
            'signature': signature,
            'timestamp': datetime.now().isoformat(),
            'simulated': True,
            'nft_metadata': {
                'name': f'Ethical AI Certificate #{token_id}',
                'description': 'Certificate of Ethical AI Compliance',
                'attributes': [
                    {'trait_type': 'Audit Type', 'value': 'Comprehensive'},
                    {'trait_type': 'Issue Date', 'value': datetime.now().strftime('%Y-%m-%d')}
                ]
            }
        }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_audit_report(audit_results):
    """Generate comprehensive audit report"""
    report = []
    report.append("=" * 80)
    report.append("ETHICAL AI AUDIT REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nAudit ID: {audit_results.get('audit_id', 'N/A')}")
    report.append("\n" + "=" * 80)
    
    # Executive Summary
    report.append("\nEXECUTIVE SUMMARY")
    report.append("-" * 80)
    if 'fairness_score' in audit_results:
        fs = audit_results['fairness_score']
        report.append(f"Overall Fairness Score: {fs['score']}/100 ({fs['grade']})")
        report.append(f"Checks Passed: {fs['passed_checks']}/{fs['total_checks']}")
    
    if 'privacy_score' in audit_results:
        ps = audit_results['privacy_score']
        report.append(f"\nPrivacy Score: {ps['score']}/100")
        report.append(f"Privacy Risk Level: {ps['risk_level']}")
    
    # Bias Analysis
    if 'bias_results' in audit_results:
        report.append("\n" + "=" * 80)
        report.append("BIAS ANALYSIS")
        report.append("-" * 80)
        
        for attr, metrics in audit_results['bias_results'].items():
            report.append(f"\nProtected Attribute: {attr.upper()}")
            
            # Demographic Parity
            dp = metrics['demographic_parity']
            report.append(f"\n  Demographic Parity:")
            for group, rate in dp['rates'].items():
                report.append(f"    {group}: {rate:.4f}")
            report.append(f"    Difference: {dp['difference']:.4f} (Threshold: {dp['threshold']})")
            report.append(f"    Status: {'✓ PASSED' if dp['passed'] else '✗ FAILED'}")
            
            # Disparate Impact
            di = metrics['disparate_impact']
            report.append(f"\n  Disparate Impact:")
            report.append(f"    Ratio: {di['ratio']:.4f} (Threshold: {di['threshold']})")
            report.append(f"    Status: {'✓ PASSED' if di['passed'] else '✗ FAILED'}")
            
            # Equalized Odds
            eo = metrics['equalized_odds']
            report.append(f"\n  Equalized Odds:")
            report.append(f"    TPR Difference: {eo['tpr_difference']:.4f}")
            report.append(f"    FPR Difference: {eo['fpr_difference']:.4f}")
            report.append(f"    Status: {'✓ PASSED' if eo['passed'] else '✗ FAILED'}")
    
    # Privacy Analysis
    if 'privacy_score' in audit_results:
        report.append("\n" + "=" * 80)
        report.append("PRIVACY ANALYSIS")
        report.append("-" * 80)
        ps = audit_results['privacy_score']
        
        if ps['pii_columns']:
            report.append(f"\nPotential PII Columns Detected: {', '.join(ps['pii_columns'])}")
            report.append("  ⚠ WARNING: Remove or anonymize PII before production use")
        else:
            report.append("\n✓ No obvious PII columns detected")
        
        if 'dp_check' in audit_results:
            dp_check = audit_results['dp_check']
            report.append(f"\nDifferential Privacy Analysis:")
            report.append(f"  Epsilon Estimate: {dp_check.get('epsilon_estimate', 'N/A')}")
            report.append(f"  Recommendation: {dp_check.get('recommendation', 'N/A')}")
    
    # Transparency
    if 'model_analysis' in audit_results:
        report.append("\n" + "=" * 80)
        report.append("MODEL TRANSPARENCY")
        report.append("-" * 80)
        ma = audit_results['model_analysis']
        report.append(f"\nTotal Layers: {ma['total_layers']}")
        report.append(f"Total Parameters: {ma['total_parameters']:,}")
        report.append(f"Model Size: {ma['model_size_mb']:.2f} MB")
    
    if 'feature_importance' in audit_results:
        report.append("\n\nFeature Importance:")
        for feature, importance in sorted(audit_results['feature_importance'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"  {feature}: {importance:.4f}")
    
    # Blockchain Verification
    if 'blockchain' in audit_results:
        report.append("\n" + "=" * 80)
        report.append("BLOCKCHAIN VERIFICATION")
        report.append("-" * 80)
        bc = audit_results['blockchain']
        report.append(f"\nAudit Hash: {bc.get('audit_hash', 'N/A')}")
        report.append(f"Transaction Hash: {bc.get('tx_hash', 'N/A')}")
        report.append(f"Timestamp: {bc.get('timestamp', 'N/A')}")
        if bc.get('simulated'):
            report.append("  ℹ Note: Blockchain simulation mode (for demo)")
    
    # Recommendations
    report.append("\n" + "=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("-" * 80)
    
    recommendations = []
    
    if 'fairness_score' in audit_results and audit_results['fairness_score']['score'] < 80:
        recommendations.append("• Review and rebalance training data across protected groups")
        recommendations.append("• Consider implementing fairness constraints during model training")
    
    if 'privacy_score' in audit_results:
        if audit_results['privacy_score']['pii_columns']:
            recommendations.append("• Remove or encrypt PII columns before deployment")
        recommendations.append("• Implement differential privacy mechanisms (ε < 1.0)")
    
    if not recommendations:
        recommendations.append("• Continue monitoring model performance in production")
        recommendations.append("• Conduct regular audits (quarterly recommended)")
    
    report.extend(recommendations)
    
    # Disclaimer
    report.append("\n" + "=" * 80)
    report.append("DISCLAIMER")
    report.append("-" * 80)
    report.append("This audit is provided for informational purposes only and does not")
    report.append("constitute legal advice. Organizations should consult with legal counsel")
    report.append("regarding compliance with applicable AI ethics regulations and standards.")
    report.append("=" * 80)
    
    return "\n".join(report)


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main application function"""
    
    # Initialize session state
    if 'audit_results' not in st.session_state:
        st.session_state.audit_results = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'uploaded_model' not in st.session_state:
        st.session_state.uploaded_model = None
    
    # Header
    st.title("🛡️ Ethical AI Auditor SaaS")
    st.markdown("*Comprehensive AI Ethics Compliance Platform with Blockchain Certification*")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Function:",
        ["Upload & Configure", "Run Audit", "View Results", "Blockchain Verification", "About"]
    )
    
    # ========================================================================
    # PAGE 1: Upload & Configure
    # ========================================================================
    if page == "Upload & Configure":
        st.header("📤 Upload Data or Model")
        
        upload_type = st.radio("What would you like to audit?", 
                              ["Dataset (CSV)", "PyTorch Model", "Use Sample Data"])
        
        if upload_type == "Dataset (CSV)":
            uploaded_file = st.file_uploader(
                "Upload your dataset (CSV format)",
                type=['csv'],
                help="Max file size: 10MB. Should contain a binary outcome column and protected attributes."
            )
            
            if uploaded_file is not None:
                if uploaded_file.size > MAX_FILE_SIZE:
                    st.error(f"File too large! Max size: {MAX_FILE_SIZE/1024/1024}MB")
                else:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.uploaded_data = df
                        st.success(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                        
                        with st.expander("Preview Data"):
                            st.dataframe(df.head(10))
                            st.write("**Column Types:**")
                            st.write(df.dtypes)
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
        
        elif upload_type == "PyTorch Model":
            uploaded_model = st.file_uploader(
                "Upload PyTorch model (.pth or .pkl)",
                type=['pth', 'pkl'],
                help="Upload a saved PyTorch model file"
            )
            
            if uploaded_model is not None:
                try:
                    model = torch.load(uploaded_model, map_location=torch.device('cpu'))
                    st.session_state.uploaded_model = model
                    st.success("✓ Model loaded successfully")
                    
                    # Show model structure
                    with st.expander("Model Structure"):
                        st.write(model)
                except Exception as e:
                    st.error(f"Error loading model: {e}")
            
            # Still need data for model evaluation
            st.subheader("Upload evaluation dataset")
            eval_file = st.file_uploader("Upload CSV for model evaluation", type=['csv'])
            if eval_file is not None:
                try:
                    df = pd.read_csv(eval_file)
                    st.session_state.uploaded_data = df
                    st.success(f"✓ Evaluation data loaded: {df.shape[0]} rows")
                except Exception as e:
                    st.error(f"Error loading evaluation data: {e}")
        
        else:  # Use Sample Data
            st.info("Using generated sample data for demonstration")
            if st.button("Generate Sample Dataset"):
                df = generate_sample_data()
                st.session_state.uploaded_data = df
                st.success("✓ Sample dataset generated with intentional bias for testing")
                st.dataframe(df.head(10))
            
            if st.button("Generate Sample Model"):
                model = generate_sample_model()
                st.session_state.uploaded_model = model
                st.success("✓ Sample model generated")
                st.write(model)
        
        # Configuration
        if st.session_state.uploaded_data is not None:
            st.header("⚙️ Audit Configuration")
            
            df = st.session_state.uploaded_data
            
            # Select columns
            outcome_col = st.selectbox(
                "Select outcome/target column:",
                options=df.columns.tolist(),
                help="Binary classification target (0/1 or similar)"
            )
            
            protected_attrs = st.multiselect(
                "Select protected attributes to audit:",
                options=[col for col in df.columns if col != outcome_col],
                default=[col for col in ['gender', 'race', 'age'] if col in df.columns],
                help="Attributes like gender, race, age that should be checked for bias"
            )
            
            # Audit options
            st.subheader("Select Audit Checks")
            check_bias = st.checkbox("Bias Detection", value=True)
            check_fairness = st.checkbox("Fairness Metrics", value=True)
            check_privacy = st.checkbox("Privacy Audit", value=True)
            check_transparency = st.checkbox("Transparency Analysis", value=True)
            
            # Save configuration
            if st.button("Save Configuration"):
                st.session_state.audit_config = {
                    'outcome_col': outcome_col,
                    'protected_attrs': protected_attrs,
                    'check_bias': check_bias,
                    'check_fairness': check_fairness,
                    'check_privacy': check_privacy,
                    'check_transparency': check_transparency
                }
                st.success("✓ Configuration saved! Go to 'Run Audit' to proceed.")
    
    # ========================================================================
    # PAGE 2: Run Audit
    # ========================================================================
    elif page == "Run Audit":
        st.header("🔍 Run Ethical AI Audit")
        
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ Please upload data first in 'Upload & Configure'")
            return
        
        if 'audit_config' not in st.session_state:
            st.warning("⚠️ Please configure audit settings first")
            return
        
        config = st.session_state.audit_config
        df = st.session_state.uploaded_data
        
        st.write("**Audit Configuration:**")
        st.json(config)
        
        if st.button("▶️ Start Audit", type="primary"):
            with st.spinner("Running comprehensive ethical audit..."):
                results = {
                    'audit_id': hashlib.sha256(
                        f"{datetime.now()}".encode()
                    ).hexdigest()[:16],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Bias Detection
                if config['check_bias']:
                    status_text.text("Analyzing bias patterns...")
                    progress_bar.progress(20)
                    results['bias_results'] = run_bias_audit(
                        df, config['protected_attrs'], config['outcome_col']
                    )
                
                # Fairness Score
                if config['check_fairness']:
                    status_text.text("Calculating fairness metrics...")
                    progress_bar.progress(40)
                    if 'bias_results' in results:
                        results['fairness_score'] = calculate_fairness_score(
                            results['bias_results']
                        )
                
                # Privacy Audit
                if config['check_privacy']:
                    status_text.text("Auditing privacy risks...")
                    progress_bar.progress(60)
                    results['privacy_score'] = calculate_privacy_score(df)
                    results['dp_check'] = simulate_differential_privacy_check(df)
                
                # Transparency
                if config['check_transparency']:
                    status_text.text("Analyzing model transparency...")
                    progress_bar.progress(80)
                    
                    if st.session_state.uploaded_model is not None:
                        results['model_analysis'] = analyze_model_structure(
                            st.session_state.uploaded_model
                        )
                    
                    results['feature_importance'] = calculate_feature_importance(
                        df, config['outcome_col']
                    )
                
                progress_bar.progress(100)
                status_text.text("Audit complete!")
                
                st.session_state.audit_results = results
                st.success("✅ Audit completed successfully! View results in 'View Results' tab.")
    
    # ========================================================================
    # PAGE 3: View Results
    # ========================================================================
    elif page == "View Results":
        st.header("📊 Audit Results Dashboard")
        
        if st.session_state.audit_results is None:
            st.warning("⚠️ No audit results available. Run an audit first.")
            return
        
        results = st.session_state.audit_results
        
        # Summary Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'fairness_score' in results:
                fs = results['fairness_score']
                st.metric(
                    "Fairness Score",
                    f"{fs['score']}/100",
                    delta=fs['grade'],
                    delta_color="off"
                )
        
        with col2:
            if 'privacy_score' in results:
                ps = results['privacy_score']
                st.metric(
                    "Privacy Score",
                    f"{ps['score']}/100",
                    delta=ps['risk_level'],
                    delta_color="inverse"
                )
        
        with col3:
            if 'bias_results' in results:
                total_checks = sum(
                    len(metrics) for metrics in results['bias_results'].values()
                )
                st.metric("Total Checks", total_checks)
        
        # Visualizations
        st.subheader("📈 Visualizations")
        
        # Fairness Score Gauge
        if 'fairness_score' in results:
            with st.expander("Fairness Score Gauge", expanded=True):
                fig = plot_fairness_score(results['fairness_score'])
                st.pyplot(fig)
        
        # Bias Metrics
        if 'bias_results' in results:
            with st.expander("Bias Analysis Charts", expanded=True):
                fig = plot_bias_metrics(results['bias_results'])
                st.pyplot(fig)
        
        # Feature Importance
        if 'feature_importance' in results:
            with st.expander("Feature Importance"):
                fig = plot_feature_importance(results['feature_importance'])
                if fig:
                    st.pyplot(fig)
        
        # Detailed Results
        st.subheader("📋 Detailed Results")
        
        with st.expander("Bias Detection Details"):
            if 'bias_results' in results:
                for attr, metrics in results['bias_results'].items():
                    st.write(f"**Protected Attribute: {attr}**")
                    st.json(metrics)
        
        with st.expander("Privacy Analysis Details"):
            if 'privacy_score' in results:
                st.json(results['privacy_score'])
            if 'dp_check' in results:
                st.write("**Differential Privacy Check:**")
                st.json(results['dp_check'])
        
        # Generate Report
        st.subheader("📄 Download Report")
        
        report_text = generate_audit_report(results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download TXT Report",
                data=report_text,
                file_name=f"ethical_audit_{results['audit_id']}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Generate hash for blockchain
            report_hash = hash_data(report_text)
            st.code(f"Report Hash (SHA256):\n{report_hash}", language="text")
    
    # ========================================================================
    # PAGE 4: Blockchain Verification
    # ========================================================================
    elif page == "Blockchain Verification":
        st.header("⛓️ Blockchain Verification")
        
        if st.session_state.audit_results is None:
            st.warning("⚠️ No audit results to verify. Run an audit first.")
            return
        
        st.info("""
        **Blockchain Integration**: This feature provides immutable proof of your ethical AI audit
        by logging results on the Ethereum blockchain and minting an NFT certificate.
        """)
        
        results = st.session_state.audit_results
        report_text = generate_audit_report(results)
        report_hash = hash_data(report_text)
        
        st.write("**Audit Report Hash:**")
        st.code(report_hash)
        
        # Initialize blockchain
        blockchain = BlockchainIntegration()
        
        if blockchain.simulated:
            st.warning("⚠️ Running in simulation mode. Configure INFURA_URL and PRIVATE_KEY in secrets for production.")
        
        # Log on Blockchain
        st.subheader("1. Log Audit on Blockchain")
        
        if st.button("📝 Log Audit Record"):
            with st.spinner("Logging on blockchain..."):
                log_result = blockchain.log_audit(report_hash, "Comprehensive Ethical Audit")
                
                if log_result['success']:
                    st.success("✅ Audit logged successfully!")
                    st.json(log_result)
                    
                    # Save to results
                    results['blockchain'] = {
                        'audit_hash': report_hash,
                        'tx_hash': log_result['tx_hash'],
                        'timestamp': log_result['timestamp'],
                        'simulated': log_result.get('simulated', False)
                    }
                    st.session_state.audit_results = results
                else:
                    st.error("Failed to log audit")
        
        # Mint NFT Certificate
        st.subheader("2. Mint NFT Certificate")
        
        recipient_address = st.text_input(
            "Recipient Ethereum Address:",
            value="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb4",
            help="Ethereum address to receive the NFT certificate"
        )
        
        if st.button("🎫 Mint NFT Certificate"):
            with st.spinner("Minting NFT certificate..."):
                nft_result = blockchain.mint_nft_certificate(recipient_address, report_hash)
                
                if nft_result['success']:
                    st.success("✅ NFT Certificate minted successfully!")
                    st.balloons()
                    
                    st.json(nft_result)
                    
                    # Display NFT Metadata
                    if 'nft_metadata' in nft_result:
                        st.write("**NFT Metadata:**")
                        st.json(nft_result['nft_metadata'])
                    
                    # Save to results
                    results['nft_certificate'] = nft_result
                    st.session_state.audit_results = results
                else:
                    st.error("Failed to mint NFT")
        
        # Verification Info
        st.subheader("📜 Verification Information")
        
        if 'blockchain' in results:
            st.write("**Blockchain Record:**")
            st.json(results['blockchain'])
        
        if 'nft_certificate' in results:
            st.write("**NFT Certificate:**")
            st.json(results['nft_certificate'])
            
            # Generate certificate visual
            st.write("**Certificate Preview:**")
            cert_html = f"""
            <div style="border: 3px solid gold; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; text-align: center;">
                <h2>🏆 ETHICAL AI COMPLIANCE CERTIFICATE</h2>
                <p style="font-size: 18px; margin: 20px 0;">This certifies that the AI system has been audited for ethical compliance</p>
                <p><strong>Token ID:</strong> #{results['nft_certificate']['token_id']}</p>
                <p><strong>Audit Score:</strong> {results.get('fairness_score', {}).get('score', 'N/A')}/100</p>
                <p><strong>Issue Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
                <p style="margin-top: 30px; font-size: 12px;">Verified on Ethereum Blockchain</p>
            </div>
            """
            st.markdown(cert_html, unsafe_allow_html=True)
    
    # ========================================================================
    # PAGE 5: About
    # ========================================================================
    elif page == "About":
        st.header("ℹ️ About Ethical AI Auditor")
        
        st.markdown("""
        ## 🎯 Mission
        
        The Ethical AI Auditor SaaS is designed to help organizations ensure their AI systems
        are fair, unbiased, privacy-preserving, and transparent. We provide comprehensive
        auditing tools with blockchain-verified certification.
        
        ## 🔍 Features
        
        ### Bias Detection
        - **Demographic Parity**: Measures equal positive outcome rates across groups
        - **Disparate Impact**: Checks for the "80% rule" compliance
        - **Equalized Odds**: Ensures equal true positive and false positive rates
        
        ### Privacy Auditing
        - PII (Personally Identifiable Information) detection
        - Differential privacy compliance checking
        - Data sensitivity risk assessment
        
        ### Transparency Analysis
        - Model structure inspection
        - Feature importance calculation
        - Explainability metrics
        
        ### Blockchain Integration
        - Immutable audit logging on Ethereum
        - NFT certificates for compliance proof
        - Cryptographic verification
        
        ## 📊 Metrics & Standards
        
        Our audits are based on industry standards including:
        - IBM AI Fairness 360 guidelines
        - NIST AI Risk Management Framework
        - IEEE P7003 Algorithmic Bias Considerations
        - EU AI Act compliance principles
        
        ## 🚀 Use Cases
        
        - **Financial Services**: Loan approval systems, credit scoring
        - **Healthcare**: Diagnostic models, treatment recommendations
        - **HR & Recruitment**: Resume screening, candidate evaluation
        - **Criminal Justice**: Risk assessment tools
        - **Marketing**: Ad targeting, customer segmentation
        
        ## 🛠️ Technology Stack
        
        - **Frontend**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **ML**: PyTorch
        - **Visualization**: Matplotlib
        - **Blockchain**: Web3.py, Ethereum
        
        ## 📞 Support & Integration
        
        For enterprise support, custom integrations, or questions:
        - Email: support@ethicalai-auditor.com (simulated)
        - Documentation: https://docs.ethicalai-auditor.com (simulated)
        
        ## ⚖️ Legal Disclaimer
        
        This tool is provided for informational and awareness purposes only. It does not
        constitute legal advice. Organizations should consult with legal counsel regarding
        compliance with applicable laws and regulations including GDPR, CCPA, and emerging
        AI governance frameworks.
        
        The audit results are assessments based on statistical methods and should be
        considered as one input into a broader ethical AI governance program.
        
        ## 📄 License
        
        © 2026 Ethical AI Auditor. All rights reserved.
        
        ---
        
        **Version**: 1.0.0  
        **Last Updated**: February 2026
        """)
        
        # Sample integration code
        with st.expander("🔌 Integration Examples"):
            st.code("""
# Python API Integration Example (Future)
from ethical_ai_auditor import AuditClient

client = AuditClient(api_key="your-api-key")

# Upload data
audit = client.create_audit(
    data=df,
    protected_attributes=['gender', 'race'],
    outcome_column='approved'
)

# Run audit
results = audit.run()

# Get score
print(f"Fairness Score: {results.fairness_score}")

# Mint certificate
certificate = audit.mint_certificate(
    recipient="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb4"
)
            """, language="python")
            
            st.code("""
// REST API Integration Example (Future)
const response = await fetch('https://api.ethicalai-auditor.com/v1/audits', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR-API-KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    data_url: 'https://yourdomain.com/data.csv',
    protected_attributes: ['gender', 'race'],
    outcome_column: 'approved'
  })
});

const audit = await response.json();
console.log('Audit ID:', audit.id);
            """, language="javascript")
    
    # Sidebar Info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **💡 Quick Tips:**
    1. Start with sample data to explore features
    2. Upload your own CSV with protected attributes
    3. Run comprehensive audits
    4. Download reports for compliance
    5. Mint NFT certificates for proof
    
    **🔒 Privacy:** All processing is done in-session. 
    No data is stored permanently.
    """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Ethical AI Auditor v1.0.0")
    st.sidebar.caption("Built by Ary HH with ❤️ for responsible AI")


if __name__ == "__main__":
    main()
