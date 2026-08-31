(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SAFPredictorCore = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function asNumber(value) {
    if (typeof value === "number") return value;
    if (typeof value === "string" && value.trim() !== "") return Number(value);
    return NaN;
  }

  function upperBound(sorted, value) {
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] <= value) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function predictRandomForest(model, x) {
    let sum = 0;
    for (const tree of model.trees) {
      let node = 0;
      while (tree.feature[node] >= 0) {
        node = Math.fround(x[tree.feature[node]]) <= tree.threshold[node]
          ? tree.children_left[node]
          : tree.children_right[node];
      }
      sum += tree.value[node];
    }
    return sum / model.trees.length;
  }

  function predictSVR(model, x) {
    const scaled = x.map((value, index) => (value - model.x_mean[index]) / model.x_scale[index]);
    let transformed = model.intercept;
    for (let i = 0; i < model.support_vectors.length; i += 1) {
      let squaredDistance = 0;
      const support = model.support_vectors[i];
      for (let j = 0; j < scaled.length; j += 1) {
        const delta = scaled[j] - support[j];
        squaredDistance += delta * delta;
      }
      transformed += model.dual_coef[i] * Math.exp(-model.gamma * squaredDistance);
    }
    return transformed * model.y_scale + model.y_mean;
  }

  function xgbTreeValue(tree, x) {
    let node = tree;
    while (node.leaf === undefined) {
      const split = String(node.split);
      const featureIndex = split[0] === "f" ? Number(split.slice(1)) : Number(split);
      const goLeft = Math.fround(x[featureIndex]) < Math.fround(node.split_condition);
      const target = goLeft ? node.yes : node.no;
      node = node.children.find((child) => child.nodeid === target);
      if (!node) throw new Error("Invalid XGBoost tree export");
    }
    return node.leaf;
  }

  function predictXGBoost(model, x) {
    let value = model.base_margin;
    for (const tree of model.trees) value += xgbTreeValue(tree, x);
    return value;
  }

  function predictModel(model, x) {
    if (model.kind === "random_forest") return predictRandomForest(model, x);
    if (model.kind === "svr_rbf") return predictSVR(model, x);
    if (model.kind === "xgboost_trees") return predictXGBoost(model, x);
    throw new Error(`Unsupported model kind: ${model.kind}`);
  }

  function createPredictor(bundle) {
    if (!bundle || bundle.schema_version !== 1) throw new Error("Unsupported SAF-Predict model bundle");
    const inputOrder = bundle.input_order.slice();
    const targetOrder = bundle.target_order.slice();
    const med = bundle.ood.feature_median;
    const scale = bundle.ood.feature_scale;
    const developmentRecords = bundle.ood.training_records;
    const referenceDistances = bundle.ood.group_reference_distances_sorted;

    function validate(row) {
      const errors = [];
      const warnings = [];
      const x = inputOrder.map((code) => asNumber(row[code]));
      inputOrder.forEach((code, index) => {
        const value = x[index];
        const meta = bundle.input_meta[code];
        const range = bundle.input_ranges[code];
        if (!Number.isFinite(value)) {
          errors.push(`${code} (${meta.name}) must be a finite numeric value.`);
          return;
        }
        if (value < range.min || value > range.max) {
          warnings.push(`${code} lies outside the observed development-set range (${range.min.toFixed(3)}–${range.max.toFixed(3)}).`);
        } else if (value < range.q05 || value > range.q95) {
          warnings.push(`${code} lies outside the central P5–P95 range of the development set.`);
        }
      });
      if (Number.isFinite(x[0]) && x[0] <= 0) errors.push("Molar mass must be greater than zero.");
      if (Number.isFinite(x[1]) && x[1] <= 0) errors.push("The H/C atomic ratio must be greater than zero.");
      if (Number.isFinite(x[2]) && (x[2] < 0 || x[2] > 1)) errors.push("The aromatic carbon fraction must lie between 0 and 1.");
      for (const index of [3, 4]) {
        if (Number.isFinite(x[index]) && (!Number.isInteger(x[index]) || x[index] < 0)) {
          errors.push(`${inputOrder[index]} must be a non-negative integer.`);
        }
      }
      if (Number.isFinite(x[5]) && x[5] < 0) errors.push("The graph-automorphism descriptor cannot be negative.");
      if (Number.isFinite(x[6]) && x[6] <= 0) errors.push("The Kier κ2 shape index must be greater than zero.");
      return { valid: errors.length === 0, errors, warnings, x };
    }

    function applicability(x) {
      const z = x.map((value, index) => (value - med[index]) / scale[index]);
      const neighbours = developmentRecords.map((record) => {
        let distance = 0;
        for (let j = 0; j < x.length; j += 1) {
          const trainZ = (record.x[j] - med[j]) / scale[j];
          distance += Math.abs(z[j] - trainZ);
        }
        return { record_id: record.record_id, distance: distance / x.length };
      }).sort((a, b) => a.distance - b.distance);
      const distance = neighbours[0].distance;
      const percentile = Math.min(100, 100 * (upperBound(referenceDistances, distance) + 1) / (referenceDistances.length + 1));
      let band = "below_90";
      let label = "Below the 90th percentile of the applicability-domain reference-distance distribution";
      if (percentile >= 95) {
        band = "at_or_above_95";
        label = "At or above the 95th percentile of the applicability-domain reference-distance distribution";
      } else if (percentile >= 90) {
        band = "90_to_below_95";
        label = "From the 90th to below the 95th percentile of the applicability-domain reference-distance distribution";
      }
      return { distance, percentile, band, label, nearest: neighbours.slice(0, 3) };
    }

    function outputWarnings(predictions) {
      const warnings = [];
      if (predictions.Y03 <= 0) warnings.push("The predicted density is non-positive; do not use this result.");
      if (predictions.Y07 <= 0) warnings.push("The predicted kinematic viscosity is non-positive; do not use this result.");
      if (predictions.Y04 >= predictions.Y05) warnings.push("The predicted solid–liquid phase-transition temperature is not below the predicted boiling point; the result fails a basic physical-consistency check.");
      if (predictions.Y05 < 20) warnings.push("The predicted boiling point is below 20 °C; the compound may not be liquid under ambient conditions.");
      if (predictions.Y04 > 40) warnings.push("The predicted solid–liquid phase-transition temperature exceeds 40 °C; interpretation of Y07 as a liquid-phase property at 40 °C may be invalid.");
      if (predictions.Y06 >= predictions.Y05) warnings.push("The predicted flash point is not below the predicted boiling point; verify the structural inputs and source conditions.");
      return warnings;
    }

    function predictOne(row) {
      const checked = validate(row);
      if (!checked.valid) return { ok: false, errors: checked.errors, warnings: checked.warnings };
      const x = checked.x;
      const predictions = {};
      for (const target of Object.keys(bundle.models)) predictions[target] = predictModel(bundle.models[target], x);
      predictions.Y02 = predictions.Y01 * predictions.Y03;

      const outputs = {};
      for (const target of targetOrder) {
        const point = predictions[target];
        const interval = bundle.intervals[target];
        const range = bundle.output_ranges[target];
        outputs[target] = {
          point,
          lower90: point - interval.half_width_90,
          upper90: point + interval.half_width_90,
          lower95: point - interval.half_width_95,
          upper95: point + interval.half_width_95,
          developmentRangePosition: Math.max(0, Math.min(100, 100 * (point - range.min) / (range.max - range.min))),
          model: bundle.deployment_models[target],
          provenance: bundle.provenance_counts[target],
        };
      }
      const ad = applicability(x);
      const warnings = checked.warnings.concat(outputWarnings(predictions));
      if (ad.band === "at_or_above_95") warnings.push("The applicability-domain (AD) distance is at or above the 95th percentile of the development-set reference-distance distribution.");
      else if (ad.band === "90_to_below_95") warnings.push("The applicability-domain (AD) distance is from the 90th to below the 95th percentile of the development-set reference-distance distribution.");
      return { ok: true, x, outputs, applicability: ad, warnings };
    }

    return {
      bundle,
      validate,
      predictOne,
      predictBatch: (rows) => rows.map((row) => predictOne(row)),
    };
  }

  return { createPredictor, predictModel };
});
