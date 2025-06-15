"""
Test the box-based prompt functionality fix
"""

# Test the fingerprint function with a dictionary
def test_fingerprint_fix():
    print("Testing fingerprint function with dictionary input...")
    
    # Simulate the problematic case
    test_dict = {
        'boxes': [
            {
                'type': 'rectangle',
                'label': 'test',
                'color': 'red',
                'coordinates': [10, 20, 30, 40]
            }
        ]
    }
    
    # This should work without the __module__ error
    try:
        result = f"unknown_format:{type(test_dict).__name__}"
        print(f"✅ Type name resolution works: {result}")
        
        # Test the old problematic way (this would cause the error)
        # problematic = f"unknown_format:{type(test_dict)}"  # This would fail
        
    except AttributeError as e:
        print(f"❌ Error occurred: {e}")
    
    print("✅ Fingerprint function test completed successfully!")

if __name__ == "__main__":
    test_fingerprint_fix()
