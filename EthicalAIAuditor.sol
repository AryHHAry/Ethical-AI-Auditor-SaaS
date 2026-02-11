// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title EthicalAIAuditor
 * @dev Smart contract for logging AI ethics audits and minting compliance certificates as NFTs
 * 
 * DEPLOYMENT INSTRUCTIONS:
 * 
 * Option 1 - Using Remix (Easiest):
 * 1. Go to https://remix.ethereum.org
 * 2. Create new file: contracts/EthicalAIAuditor.sol
 * 3. Paste this code
 * 4. Install OpenZeppelin:
 *    - In Remix, go to Plugin Manager
 *    - Activate "OPENZEPPELIN CONTRACTS" plugin
 * 5. Compile:
 *    - Select compiler version 0.8.0 or higher
 *    - Click "Compile EthicalAIAuditor.sol"
 * 6. Deploy:
 *    - Environment: "Injected Provider - MetaMask"
 *    - Network: Sepolia Testnet
 *    - Click "Deploy"
 *    - Confirm transaction in MetaMask
 * 7. Copy deployed contract address
 * 
 * Option 2 - Using Hardhat:
 * 1. npm install --save-dev hardhat @openzeppelin/contracts
 * 2. npx hardhat init
 * 3. Copy this file to contracts/
 * 4. Create deployment script (see below)
 * 5. npx hardhat run scripts/deploy.js --network sepolia
 * 
 * REQUIRED SETUP:
 * - MetaMask with Sepolia testnet configured
 * - Sepolia ETH from faucet: https://sepoliafaucet.com
 * - Infura account (for backend integration)
 */

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @dev Main contract for Ethical AI Auditor
 * Features:
 * - Log audit records immutably on-chain
 * - Mint ERC-721 NFT certificates for compliance proof
 * - Query historical audit records
 * - Transfer certificates to new owners
 */
contract EthicalAIAuditor is ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    
    // Token ID counter for NFTs
    Counters.Counter private _tokenIdCounter;
    
    /**
     * @dev Structure to store audit record details
     */
    struct AuditRecord {
        string reportHash;        // SHA256 hash of full audit report
        uint256 timestamp;        // Block timestamp when logged
        address auditor;          // Address that logged the audit
        string auditType;         // Type of audit (e.g., "Comprehensive", "Bias Only")
        uint256 fairnessScore;    // Fairness score (0-100)
        uint256 privacyScore;     // Privacy score (0-100)
        bool certified;           // Whether an NFT certificate was minted
    }
    
    /**
     * @dev Mapping from audit hash to audit record
     */
    mapping(bytes32 => AuditRecord) public auditRecords;
    
    /**
     * @dev Mapping from token ID to audit hash
     * Links NFT certificates to their corresponding audits
     */
    mapping(uint256 => bytes32) public tokenToAudit;
    
    /**
     * @dev Mapping from audit hash to token ID
     * Reverse lookup to check if audit has certificate
     */
    mapping(bytes32 => uint256) public auditToToken;
    
    /**
     * @dev Array of all audit hashes for enumeration
     */
    bytes32[] public allAudits;
    
    /**
     * @dev Events
     */
    event AuditLogged(
        bytes32 indexed auditHash,
        string reportHash,
        address indexed auditor,
        uint256 timestamp,
        string auditType,
        uint256 fairnessScore,
        uint256 privacyScore
    );
    
    event CertificateMinted(
        uint256 indexed tokenId,
        bytes32 indexed auditHash,
        address indexed recipient,
        string tokenURI
    );
    
    event CertificateTransferred(
        uint256 indexed tokenId,
        address indexed from,
        address indexed to
    );
    
    /**
     * @dev Constructor
     */
    constructor() ERC721("Ethical AI Certificate", "EAIC") {
        // Token name: "Ethical AI Certificate"
        // Token symbol: "EAIC"
    }
    
    /**
     * @dev Log a new audit record
     * @param _reportHash SHA256 hash of the complete audit report
     * @param _auditType Type of audit conducted
     * @param _fairnessScore Fairness score from 0-100
     * @param _privacyScore Privacy score from 0-100
     * @return auditHash Unique identifier for this audit
     */
    function logAudit(
        string memory _reportHash,
        string memory _auditType,
        uint256 _fairnessScore,
        uint256 _privacyScore
    ) public returns (bytes32) {
        require(_fairnessScore <= 100, "Fairness score must be <= 100");
        require(_privacyScore <= 100, "Privacy score must be <= 100");
        
        // Generate unique audit hash
        bytes32 auditHash = keccak256(
            abi.encodePacked(
                _reportHash,
                block.timestamp,
                msg.sender,
                _auditType
            )
        );
        
        // Ensure audit doesn't already exist
        require(auditRecords[auditHash].timestamp == 0, "Audit already exists");
        
        // Store audit record
        auditRecords[auditHash] = AuditRecord({
            reportHash: _reportHash,
            timestamp: block.timestamp,
            auditor: msg.sender,
            auditType: _auditType,
            fairnessScore: _fairnessScore,
            privacyScore: _privacyScore,
            certified: false
        });
        
        // Add to enumeration array
        allAudits.push(auditHash);
        
        // Emit event
        emit AuditLogged(
            auditHash,
            _reportHash,
            msg.sender,
            block.timestamp,
            _auditType,
            _fairnessScore,
            _privacyScore
        );
        
        return auditHash;
    }
    
    /**
     * @dev Mint an NFT certificate for a logged audit
     * @param to Address to receive the certificate
     * @param auditHash Hash of the audit to certify
     * @param tokenURI Metadata URI for the NFT (IPFS or HTTP)
     * @return tokenId ID of the minted NFT
     */
    function mintCertificate(
        address to,
        bytes32 auditHash,
        string memory tokenURI
    ) public onlyOwner returns (uint256) {
        require(to != address(0), "Cannot mint to zero address");
        require(auditRecords[auditHash].timestamp != 0, "Audit does not exist");
        require(!auditRecords[auditHash].certified, "Certificate already minted");
        
        // Get current token ID and increment
        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();
        
        // Mint NFT
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI);
        
        // Link NFT to audit
        tokenToAudit[tokenId] = auditHash;
        auditToToken[auditHash] = tokenId;
        
        // Mark audit as certified
        auditRecords[auditHash].certified = true;
        
        // Emit event
        emit CertificateMinted(tokenId, auditHash, to, tokenURI);
        
        return tokenId;
    }
    
    /**
     * @dev Get audit record details
     * @param auditHash Hash of the audit
     * @return Audit record struct
     */
    function getAuditRecord(bytes32 auditHash) 
        public 
        view 
        returns (AuditRecord memory) 
    {
        require(auditRecords[auditHash].timestamp != 0, "Audit does not exist");
        return auditRecords[auditHash];
    }
    
    /**
     * @dev Get audit hash for a certificate token
     * @param tokenId ID of the NFT certificate
     * @return auditHash Hash of the associated audit
     */
    function getAuditForToken(uint256 tokenId) 
        public 
        view 
        returns (bytes32) 
    {
        require(_exists(tokenId), "Token does not exist");
        return tokenToAudit[tokenId];
    }
    
    /**
     * @dev Get certificate token ID for an audit
     * @param auditHash Hash of the audit
     * @return tokenId ID of the NFT certificate (0 if not minted)
     */
    function getTokenForAudit(bytes32 auditHash) 
        public 
        view 
        returns (uint256) 
    {
        return auditToToken[auditHash];
    }
    
    /**
     * @dev Get total number of audits logged
     * @return count Total audit count
     */
    function getTotalAudits() public view returns (uint256) {
        return allAudits.length;
    }
    
    /**
     * @dev Get audit hash by index (for enumeration)
     * @param index Index in the audits array
     * @return auditHash Hash of the audit
     */
    function getAuditByIndex(uint256 index) 
        public 
        view 
        returns (bytes32) 
    {
        require(index < allAudits.length, "Index out of bounds");
        return allAudits[index];
    }
    
    /**
     * @dev Check if an audit has been certified
     * @param auditHash Hash of the audit
     * @return certified Whether certificate was minted
     */
    function isCertified(bytes32 auditHash) 
        public 
        view 
        returns (bool) 
    {
        return auditRecords[auditHash].certified;
    }
    
    /**
     * @dev Override transfer function to emit custom event
     */
    function _transfer(
        address from,
        address to,
        uint256 tokenId
    ) internal virtual override {
        super._transfer(from, to, tokenId);
        emit CertificateTransferred(tokenId, from, to);
    }
    
    /**
     * @dev Get all audits by a specific auditor
     * Note: This is gas-intensive for large datasets, use off-chain indexing for production
     * @param auditor Address of the auditor
     * @return Array of audit hashes
     */
    function getAuditsByAuditor(address auditor) 
        public 
        view 
        returns (bytes32[] memory) 
    {
        // Count audits by this auditor
        uint256 count = 0;
        for (uint256 i = 0; i < allAudits.length; i++) {
            if (auditRecords[allAudits[i]].auditor == auditor) {
                count++;
            }
        }
        
        // Create result array
        bytes32[] memory result = new bytes32[](count);
        uint256 resultIndex = 0;
        
        // Populate result array
        for (uint256 i = 0; i < allAudits.length; i++) {
            if (auditRecords[allAudits[i]].auditor == auditor) {
                result[resultIndex] = allAudits[i];
                resultIndex++;
            }
        }
        
        return result;
    }
}

/**
 * HARDHAT DEPLOYMENT SCRIPT (scripts/deploy.js)
 * 
 * const hre = require("hardhat");
 * 
 * async function main() {
 *   console.log("Deploying EthicalAIAuditor contract...");
 *   
 *   const EthicalAIAuditor = await hre.ethers.getContractFactory("EthicalAIAuditor");
 *   const auditor = await EthicalAIAuditor.deploy();
 *   
 *   await auditor.deployed();
 *   
 *   console.log("EthicalAIAuditor deployed to:", auditor.address);
 *   console.log("Save this address for your app configuration!");
 * }
 * 
 * main()
 *   .then(() => process.exit(0))
 *   .catch((error) => {
 *     console.error(error);
 *     process.exit(1);
 *   });
 * 
 * 
 * HARDHAT CONFIG (hardhat.config.js)
 * 
 * require("@nomiclabs/hardhat-waffle");
 * require("dotenv").config();
 * 
 * module.exports = {
 *   solidity: "0.8.0",
 *   networks: {
 *     sepolia: {
 *       url: process.env.INFURA_URL,
 *       accounts: [process.env.PRIVATE_KEY]
 *     }
 *   }
 * };
 * 
 * 
 * USAGE FROM PYTHON (app.py integration)
 * 
 * from web3 import Web3
 * import json
 * 
 * # Connect to network
 * web3 = Web3(Web3.HTTPProvider(INFURA_URL))
 * account = web3.eth.account.from_key(PRIVATE_KEY)
 * 
 * # Load contract ABI and address
 * contract = web3.eth.contract(
 *     address=CONTRACT_ADDRESS,
 *     abi=CONTRACT_ABI
 * )
 * 
 * # Log audit
 * tx = contract.functions.logAudit(
 *     report_hash,
 *     "Comprehensive",
 *     fairness_score,
 *     privacy_score
 * ).build_transaction({
 *     'from': account.address,
 *     'nonce': web3.eth.get_transaction_count(account.address),
 *     'gas': 200000,
 *     'gasPrice': web3.eth.gas_price
 * })
 * 
 * signed_tx = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
 * tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
 * receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
 * 
 * # Get audit hash from event
 * audit_hash = contract.events.AuditLogged().process_receipt(receipt)[0]['args']['auditHash']
 * 
 * # Mint certificate
 * tx = contract.functions.mintCertificate(
 *     recipient_address,
 *     audit_hash,
 *     token_uri
 * ).build_transaction({...})
 * 
 * signed_tx = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
 * tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
 */
