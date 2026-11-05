# Sample test questions with ground truth

Use these to evaluate retrieval accuracy, grounding and hallucination control.
"Source" is the document the answer must come from.

## Single-document factual
1. How many days of annual leave does an employee get per year? -> 24 days. (HR-POL-001 PDF)
2. How many annual leave days can be carried forward? -> Up to 10 days. (HR-POL-001 PDF)
3. How many days a week can employees work remotely, and who is ineligible? -> Up to 2 days per week with manager approval; employees in the 6-month probation period are not eligible. (HR-POL-001 PDF)
4. What is the notice period after confirmation? -> 60 days; 90 days for senior managers and above. (HR-POL-001 PDF)
5. What is the hotel cap per night in a metro city for grades L1 to L5? -> INR 6,000. (FIN-POL-007 DOCX)
6. Within how many days must an expense claim be submitted? -> Within 30 days of the end of the trip. (FIN-POL-007 DOCX)
7. Which department had the largest budget overrun in Q3, and by how much? -> Operations, INR 2,20,000 over budget (actual 37,20,000 vs budget 35,00,000). (XLSX)
8. Which invoices are marked Disputed? -> INV-2026-0466 from Vista Print and Media (INR 1,12,400 before GST). (CSV)
9. How long is the initial term of the Techspan agreement and how much notice is needed for termination for convenience? -> 24 months; 90 days written notice. (MSA RTF)
10. What is the liability cap in the Techspan agreement? -> INR 50,00,000 in any contract year. (MSA RTF)
11. How long do confidentiality obligations last under the Orbit NDA? -> 5 years after expiry or termination (trade secrets as long as they qualify). (NDA TXT)
12. Which court has jurisdiction under the Orbit NDA? -> The courts at Coimbatore. (NDA TXT)

## Image / OCR
13. What is the total payable on invoice INV-2026-0417 including GST? -> INR 57,017.60. (Scanned invoice PNG)
14. Who is the vendor on invoice INV-2026-0417 and what is the PO reference? -> Kaveri Office Supplies; PO-OPS-2026-0188. (Scanned invoice PNG)
15. What hotel rate per night is on the scanned travel expense claim? -> INR 6,400. (Scanned JPG)

## Multi-document reasoning
16. Does the scanned travel expense claim comply with the travel policy? -> No. The rate of INR 6,400 per night exceeds the INR 6,000 metro cap for grade L3, and no prior exception approval is attached. Reimbursable hotel amount is capped at 2 x 6,000 = INR 12,000 (excess INR 800 borne by the employee). The manager's approval does not replace Finance Head exception approval. (JPG + DOCX)
17. Has invoice INV-2026-0417 been paid, and when? -> Yes, paid on 2026-09-19, which is within the 30-day payment terms printed on the invoice. (CSV + PNG)
18. Is the Techspan payment timing in the MSA consistent with how the CSV shows Techspan invoices were handled? -> MSA says 45 days from receipt of a valid invoice. INV-2026-0453 (12 Sep) was paid 10 Oct (28 days). INV-2026-0491 (30 Sep) is still pending. (MSA RTF + CSV)
19. Did Operations stay within budget and what does the finance note say caused the overspend? -> No, 6.3 percent over; due to unplanned equipment maintenance (INR 1.4 lakh) and overtime in the August peak. Note: Orbit Machine Works invoice INV-2026-0485 for INR 1,40,000 equipment maintenance is in the CSV. (XLSX + CSV)
20. Does a budget revision for Operations require extra approval? -> Yes. Variance exceeds 5 percent, so approval from the Finance Head and the Managing Director is required. (XLSX Notes)

## Unanswerable / must-refuse (hallucination traps)
21. What is the company's policy on stock options? -> Not in the documents. The system should say it does not have this information.
22. What is the laptop refresh cycle? -> Not in the documents.
23. What was the Q4 FY2026 revenue? -> Not in the documents (only Q3 budget vs actual spend data exists).
24. What is the paternity leave in the Techspan contract? -> Wrong-document trap. Paternity leave (10 working days) is in the HR policy, not in the contract.
