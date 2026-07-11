from streamlit.testing.v1 import AppTest

def test_app():
    at = AppTest.from_file("app.py")
    # Streamlit requires a default timeout for running
    at.run(timeout=30)
    
    if at.exception:
        print(f"App threw exception on load: {at.exception}")
        return
        
    predict_button = None
    for b in at.button:
        if b.label == "Predict Attrition":
            predict_button = b
            break
            
    if predict_button is None:
        print("Could not find Predict button")
        return
    
    print("Found predict button. Simulating click...")
    predict_button.click()
    at.run(timeout=30)
    
    if at.exception:
        print(f"App threw exception after clicking predict: {at.exception}")
        return
        
    print("Test passed successfully! Output:")
    for text in at.error:
        print("ERROR UI TEXT:", text.value)
    for text in at.success:
        print("SUCCESS UI TEXT:", text.value)

if __name__ == "__main__":
    test_app()
