def extract_features(file_path):
    # Add your feature extraction code here
    features = extract_features_from_wav(file_path)  # Example function

    # Print out the shape of the extracted features to debug
    print(f"Extracted features shape: {features.shape}")
    return features