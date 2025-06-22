#!/usr/bin/env python3
"""
Test script for VLM checkbox mutual exclusivity logic
"""

def handle_anomalies_checkbox_change(anomalies_checked, describe_checked):
    """Handle anomalies checkbox change - ensure mutual exclusivity"""
    if anomalies_checked:
        # When anomalies is checked, uncheck describe
        return True, False  # (anomalies=True, describe=False)
    else:
        # If unchecking anomalies and describe is not checked, default to describe
        if not describe_checked:
            return False, True  # (anomalies=False, describe=True)
        return False, describe_checked  # (anomalies=False, describe=current_value)

def handle_describe_checkbox_change(describe_checked, anomalies_checked):
    """Handle describe checkbox change - ensure mutual exclusivity"""
    if describe_checked:
        # When describe is checked, uncheck anomalies
        return True, False  # (describe=True, anomalies=False)
    else:
        # If unchecking describe and anomalies is not checked, default to anomalies
        if not anomalies_checked:
            return False, True  # (describe=False, anomalies=True)
        return describe_checked, False  # (describe=current_value, anomalies=False)

def test_checkbox_logic():
    """Test the checkbox mutual exclusivity logic"""
    print("Testing VLM Checkbox Logic")
    print("=" * 40)
    
    # Test 1: Check anomalies when describe is false
    print("Test 1: Check anomalies when describe is false")
    result = handle_anomalies_checkbox_change(True, False)
    print(f"Result: anomalies={result[0]}, describe={result[1]}")
    assert result == (True, False), f"Expected (True, False), got {result}"
    print("✓ PASS\n")
    
    # Test 2: Check anomalies when describe is true
    print("Test 2: Check anomalies when describe is true")
    result = handle_anomalies_checkbox_change(True, True)
    print(f"Result: anomalies={result[0]}, describe={result[1]}")
    assert result == (True, False), f"Expected (True, False), got {result}"
    print("✓ PASS\n")
    
    # Test 3: Uncheck anomalies when describe is false
    print("Test 3: Uncheck anomalies when describe is false")
    result = handle_anomalies_checkbox_change(False, False)
    print(f"Result: anomalies={result[0]}, describe={result[1]}")
    assert result == (False, True), f"Expected (False, True), got {result}"
    print("✓ PASS\n")
    
    # Test 4: Uncheck anomalies when describe is true
    print("Test 4: Uncheck anomalies when describe is true")
    result = handle_anomalies_checkbox_change(False, True)
    print(f"Result: anomalies={result[0]}, describe={result[1]}")
    assert result == (False, True), f"Expected (False, True), got {result}"
    print("✓ PASS\n")
    
    # Test 5: Check describe when anomalies is false
    print("Test 5: Check describe when anomalies is false")
    result = handle_describe_checkbox_change(True, False)
    print(f"Result: describe={result[0]}, anomalies={result[1]}")
    assert result == (True, False), f"Expected (True, False), got {result}"
    print("✓ PASS\n")
    
    # Test 6: Check describe when anomalies is true
    print("Test 6: Check describe when anomalies is true")
    result = handle_describe_checkbox_change(True, True)
    print(f"Result: describe={result[0]}, anomalies={result[1]}")
    assert result == (True, False), f"Expected (True, False), got {result}"
    print("✓ PASS\n")
    
    # Test 7: Uncheck describe when anomalies is false
    print("Test 7: Uncheck describe when anomalies is false")
    result = handle_describe_checkbox_change(False, False)
    print(f"Result: describe={result[0]}, anomalies={result[1]}")
    assert result == (False, True), f"Expected (False, True), got {result}"
    print("✓ PASS\n")
    
    # Test 8: Uncheck describe when anomalies is true
    print("Test 8: Uncheck describe when anomalies is true")
    result = handle_describe_checkbox_change(False, True)
    print(f"Result: describe={result[0]}, anomalies={result[1]}")
    assert result == (False, False), f"Expected (False, False), got {result}"
    print("✓ PASS\n")
    
    print("All tests passed! ✓")

if __name__ == "__main__":
    test_checkbox_logic()
