import express from "express";
import bodyParser from "body-parser";
import { PythonShell } from "python-shell";
import AWS from "aws-sdk";
import uuid from "uuid";

const app = express();

app.use(bodyParser.json());

// --- S3 Setup ---
const S3_BUCKET = process.env.DIABETES_BUCKET_NAME;

let s3 = null;
if (S3_BUCKET) {
  s3 = new AWS.S3();
}

// --- Logging Function ---
function logToS3(input_data, prediction) {
  if (!s3 || !S3_BUCKET) return;

  const record = {
    timestamp: new Date().toISOString(),
    input: input_data,
    prediction: prediction,
  };

  const key = `predictions/${
    new Date().toISOString().split("T")[0]
  }/${uuid.v4()}.json`;

  s3.putObject({
    Bucket: S3_BUCKET,
    Key: key,
    Body: JSON.stringify(record),
    ContentType: "application/json",
  })
    .promise()
    .catch(console.error);
}

// --------------------
//     POST /predict
// --------------------
app.post("/predict", (req, res) => {
  const patient = req.body;

  const options = {
    args: [JSON.stringify(patient)],
    pythonOptions: ["-u"],
  };

  PythonShell.run("predict.py", options)
    .then((result) => {
      const output = JSON.parse(result[0]);

      // Log to S3
      logToS3(patient, output);

      return res.json(output);
    })
    .catch((err) => {
      console.error(err);
      return res.status(500).json({ error: "Prediction failed", details: err });
    });
});

// --- Start Server ---
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
