"""A script to recognize hand-written digits using DBN.

Author: Nahom Abi
"""

print("[X] downloading data...")
dataset = datasets.fetch_mldata("MNIST Original")

(trainX, testX, trainY, testY) = train_test_split(datasets.data / 255.0, datasets.target.astype("int0"), test_size=0.33)

dbn = DBN(
    [trainX.shape[1], 300, 10],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=1
)
dbn.fit(trainX, trainY)

preds = dbn.predict(testX)
print(classification_report(testY, preds))

for i in np.random.choice(np.arange(0, len(testY)), size=(10,)):
    pred = dbn.predict(np.atleast_2d(testX[i]))
    image = (testX[i] * 255).reshape((28, 28)).astype("uint8")

    print("Actual digit is {0}, predicted {1}".format(testY[i], pred[0]))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)
