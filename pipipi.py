from tensorflow.keras.models import load_model
model = load_model("mymodel.h5")
print(model.input_shape)