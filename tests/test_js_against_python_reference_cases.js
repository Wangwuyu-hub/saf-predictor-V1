const path = require("path");

const bundlePath = process.env.SAF_MODEL_BUNDLE_PATH
  ? path.resolve(process.env.SAF_MODEL_BUNDLE_PATH)
  : path.join(__dirname, "..", "model_bundle.js");
const bundle = require(bundlePath);
const core = require(path.join(__dirname, "..", "predictor_core.js"));
const reference = require(path.join(__dirname, "python_reference_predictions_6cases.json"));

const developmentRecords = bundle.ood.training_records;
if (developmentRecords.length !== 300) throw new Error(`Expected 300 development vectors, found ${developmentRecords.length}`);
const developmentIds = new Set();
for (const [index, record] of developmentRecords.entries()) {
  const expectedId = `DEV-${String(index + 1).padStart(4, "0")}`;
  const keys = Object.keys(record).sort().join(",");
  if (keys !== "record_id,x") throw new Error(`${expectedId}: unexpected public record keys: ${keys}`);
  if (record.record_id !== expectedId) throw new Error(`${expectedId}: surrogate ID order mismatch`);
  if (!Array.isArray(record.x) || record.x.length !== 7 || record.x.some((value) => !Number.isFinite(value))) {
    throw new Error(`${expectedId}: descriptor vector must contain seven finite values`);
  }
  developmentIds.add(record.record_id);
}
if (developmentIds.size !== 300) throw new Error("Development surrogate IDs are not unique");
for (const metadata of [bundle.input_meta, bundle.output_meta]) {
  for (const entry of Object.values(metadata)) {
    if (Object.hasOwn(entry, "name_zh")) throw new Error("Bilingual display metadata remains in the public bundle");
  }
}

const predictor = core.createPredictor(bundle);
let maximumDifference = 0;
for (const row of reference) {
  const input = Object.fromEntries(bundle.input_order.map((code, index) => [code, row.x[index]]));
  const result = predictor.predictOne(input);
  if (!result.ok) throw new Error(`Input check failed for ${row.id}: ${result.errors.join("; ")}`);
  for (const neighbour of result.applicability.nearest) {
    if (Object.keys(neighbour).sort().join(",") !== "distance,record_id") {
      throw new Error(`${row.id}: applicability output exposes unexpected development-record fields`);
    }
  }
  for (const target of bundle.target_order) {
    const difference = Math.abs(result.outputs[target].point - row.predictions[target]);
    maximumDifference = Math.max(maximumDifference, difference);
    const tolerance = target === "Y04" ? 2e-4 : 1e-8;
    if (difference > tolerance) {
      throw new Error(`${row.id}/${target}: JS ${result.outputs[target].point} vs Python ${row.predictions[target]} (Δ=${difference})`);
    }
  }
  const identityDifference = Math.abs(result.outputs.Y02.point - result.outputs.Y01.point * result.outputs.Y03.point);
  if (identityDifference > 1e-12) throw new Error(`Y02 identity failed for ${row.id}`);
}
console.log(JSON.stringify({
  sixCaseReferenceAgreementPassed: true,
  referenceCases: reference.length,
  targets: bundle.target_order.length,
  maximumDifference,
  comparison: "JavaScript predictions against six stored Python reference cases",
  publicDevelopmentRecordChecks: {
    records: developmentRecords.length,
    uniqueSurrogateIds: developmentIds.size,
    retainedFields: ["record_id", "x"],
    bilingualDisplayMetadataRemoved: true,
  },
}, null, 2));
