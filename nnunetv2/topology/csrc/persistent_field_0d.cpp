#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Pair0D {
    uint32_t birth_flat;
    int64_t death_flat;
    bool essential;
};

struct UnionFind {
    std::vector<uint32_t> parent;
    std::vector<uint32_t> birth_owner;
    std::vector<float> birth_value;

    explicit UnionFind(const std::vector<float>& values)
        : parent(values.size()), birth_owner(values.size()), birth_value(values) {
        std::iota(parent.begin(), parent.end(), uint32_t{0});
        std::iota(birth_owner.begin(), birth_owner.end(), uint32_t{0});
    }

    uint32_t find(uint32_t vertex) {
        uint32_t root = vertex;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[vertex] != vertex) {
            const uint32_t next = parent[vertex];
            parent[vertex] = root;
            vertex = next;
        }
        return root;
    }

    uint32_t link(uint32_t root_a, uint32_t root_b) {
        if (root_a == root_b) {
            return birth_owner[root_a];
        }
        const uint32_t owner_a = birth_owner[root_a];
        const uint32_t owner_b = birth_owner[root_b];
        const bool a_is_elder =
            birth_value[owner_a] > birth_value[owner_b] ||
            (birth_value[owner_a] == birth_value[owner_b] && owner_a < owner_b);
        const uint32_t survivor = a_is_elder ? root_a : root_b;
        const uint32_t killed = a_is_elder ? root_b : root_a;
        const uint32_t killed_owner = birth_owner[killed];
        parent[killed] = survivor;
        birth_owner[survivor] = a_is_elder ? owner_a : owner_b;
        return killed_owner;
    }
};

struct FieldView {
    const float* values;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t size;
};

struct PairResult {
    std::vector<Pair0D> pairs;
    std::unordered_map<uint32_t, int64_t> pair_by_birth;
};

struct MatchingResult {
    PairResult prediction;
    PairResult target;
    std::vector<int64_t> matched_prediction;
    std::vector<int64_t> matched_target;
    std::vector<int64_t> unmatched_prediction;
    std::vector<int64_t> unmatched_target;
};

struct MatchingProfile {
    double prediction_pairs_seconds;
    double target_pairs_seconds;
    double comparison_matching_seconds;
    double total_seconds;
};

FieldView validate_field(const py::array& array, const std::string& name) {
    if (!py::isinstance<py::array_t<float>>(array)) {
        throw py::type_error(name + " must have dtype float32");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(name + " must be C-contiguous");
    }
    const py::buffer_info info = array.request();
    if (info.ndim != 3) {
        throw py::value_error(name + " must be three-dimensional");
    }
    if (info.size <= 0) {
        throw py::value_error(name + " must be nonempty");
    }
    if (static_cast<uint64_t>(info.size) >= (uint64_t{1} << 32)) {
        throw py::value_error(name + " must contain fewer than 2^32 voxels");
    }
    for (const py::ssize_t dimension : info.shape) {
        if (dimension <= 0 ||
            static_cast<uint64_t>(dimension) > std::numeric_limits<uint32_t>::max()) {
            throw py::value_error(name + " has an invalid shape");
        }
    }
    const auto* values = static_cast<const float*>(info.ptr);
    for (py::ssize_t index = 0; index < info.size; ++index) {
        if (!std::isfinite(values[index])) {
            throw py::value_error(name + " must contain only finite values");
        }
    }
    return {
        values,
        static_cast<uint32_t>(info.shape[0]),
        static_cast<uint32_t>(info.shape[1]),
        static_cast<uint32_t>(info.shape[2]),
        static_cast<uint32_t>(info.size),
    };
}

std::vector<float> copy_values(const FieldView& field) {
    return std::vector<float>(field.values, field.values + field.size);
}

std::vector<uint32_t> descending_order_reference(const std::vector<float>& values) {
    std::vector<uint32_t> order(values.size());
    std::iota(order.begin(), order.end(), uint32_t{0});
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (values[a] != values[b]) {
            return values[a] > values[b];
        }
        return a < b;
    });
    return order;
}

uint32_t descending_float_key(float value) {
    if (value == 0.0F) {
        value = 0.0F;
    }
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t ascending_key =
        (bits & 0x80000000U) != 0U ? ~bits : (bits ^ 0x80000000U);
    return ~ascending_key;
}

std::vector<uint32_t> descending_order_radix(const std::vector<float>& values) {
    std::vector<uint32_t> order(values.size());
    std::iota(order.begin(), order.end(), uint32_t{0});
    std::vector<uint32_t> scratch(values.size());
    std::vector<uint32_t> keys(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        keys[index] = descending_float_key(values[index]);
    }
    for (uint32_t shift = 0; shift < 32; shift += 8) {
        std::array<size_t, 256> counts{};
        for (const uint32_t index : order) {
            ++counts[(keys[index] >> shift) & 0xffU];
        }
        std::array<size_t, 256> positions{};
        for (size_t bucket = 1; bucket < positions.size(); ++bucket) {
            positions[bucket] = positions[bucket - 1] + counts[bucket - 1];
        }
        for (const uint32_t index : order) {
            const uint32_t bucket = (keys[index] >> shift) & 0xffU;
            scratch[positions[bucket]++] = index;
        }
        order.swap(scratch);
    }
    return order;
}

std::vector<uint32_t> descending_order(const std::vector<float>& values) {
    return descending_order_radix(values);
}

template <class Function>
void for_each_neighbor(uint32_t vertex, const FieldView& field, Function&& function) {
    const uint32_t plane = field.height * field.width;
    const int64_t z = vertex / plane;
    const int64_t remainder = vertex % plane;
    const int64_t y = remainder / field.width;
    const int64_t x = remainder % field.width;
    if (z > 0 && z + 1 < field.depth && y > 0 && y + 1 < field.height &&
        x > 0 && x + 1 < field.width) {
        for (int64_t dz = -1; dz <= 1; ++dz) {
            for (int64_t dy = -1; dy <= 1; ++dy) {
                for (int64_t dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) {
                        continue;
                    }
                    const int64_t offset =
                        (dz * field.height + dy) * field.width + dx;
                    function(static_cast<uint32_t>(
                        static_cast<int64_t>(vertex) + offset));
                }
            }
        }
        return;
    }
    for (int64_t dz = -1; dz <= 1; ++dz) {
        for (int64_t dy = -1; dy <= 1; ++dy) {
            for (int64_t dx = -1; dx <= 1; ++dx) {
                if (dz == 0 && dy == 0 && dx == 0) {
                    continue;
                }
                const int64_t nz = z + dz;
                const int64_t ny = y + dy;
                const int64_t nx = x + dx;
                if (nz < 0 || ny < 0 || nx < 0 ||
                    nz >= field.depth || ny >= field.height || nx >= field.width) {
                    continue;
                }
                const uint32_t neighbor = static_cast<uint32_t>(
                    (nz * field.height + ny) * field.width + nx);
                function(neighbor);
            }
        }
    }
}

PairResult compute_pairs(const FieldView& field) {
    const std::vector<float> values = copy_values(field);
    const std::vector<uint32_t> order = descending_order(values);
    UnionFind union_find(values);
    std::vector<uint8_t> active(field.size, uint8_t{0});
    PairResult result;
    result.pairs.reserve(field.size / 4 + 1);

    for (const uint32_t vertex : order) {
        active[vertex] = uint8_t{1};
        for_each_neighbor(vertex, field, [&](uint32_t neighbor) {
            if (!active[neighbor]) {
                return;
            }
            const uint32_t root_vertex = union_find.find(vertex);
            const uint32_t root_neighbor = union_find.find(neighbor);
            if (root_vertex == root_neighbor) {
                return;
            }
            const uint32_t killed_owner = union_find.link(root_vertex, root_neighbor);
            if (values[killed_owner] > values[vertex]) {
                const int64_t pair_index = static_cast<int64_t>(result.pairs.size());
                result.pairs.push_back({killed_owner, static_cast<int64_t>(vertex), false});
                result.pair_by_birth.emplace(killed_owner, pair_index);
            }
        });
    }
    const uint32_t essential_owner = union_find.birth_owner[union_find.find(0)];
    const int64_t essential_index = static_cast<int64_t>(result.pairs.size());
    result.pairs.push_back({essential_owner, -1, true});
    result.pair_by_birth.emplace(essential_owner, essential_index);
    return result;
}

MatchingResult compute_induced_matching(
    const FieldView& prediction_field,
    const FieldView& target_field,
    MatchingProfile* profile = nullptr,
    const PairResult* precomputed_target = nullptr) {
    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();
    MatchingResult result;
    const auto prediction_start = Clock::now();
    result.prediction = compute_pairs(prediction_field);
    const auto prediction_end = Clock::now();
    if (precomputed_target == nullptr) {
        result.target = compute_pairs(target_field);
    } else {
        result.target = *precomputed_target;
    }
    const auto target_end = Clock::now();
    const std::vector<float> prediction = copy_values(prediction_field);
    const std::vector<float> target = copy_values(target_field);
    std::vector<float> comparison(prediction.size());
    std::transform(
        prediction.begin(), prediction.end(), target.begin(), comparison.begin(),
        [](float a, float b) { return std::max(a, b); });
    const std::vector<uint32_t> order = descending_order(comparison);
    UnionFind comparison_union(comparison);
    UnionFind prediction_union(prediction);
    UnionFind target_union(target);
    std::vector<uint8_t> active(comparison.size(), uint8_t{0});

    for (const uint32_t vertex : order) {
        active[vertex] = uint8_t{1};
        for_each_neighbor(vertex, prediction_field, [&](uint32_t neighbor) {
            if (!active[neighbor]) {
                return;
            }
            const uint32_t comparison_a = comparison_union.find(vertex);
            const uint32_t comparison_b = comparison_union.find(neighbor);
            if (comparison_a == comparison_b) {
                return;
            }
            const uint32_t killed_comparison =
                comparison_union.link(comparison_a, comparison_b);
            const uint32_t killed_prediction = prediction_union.link(
                prediction_union.find(vertex), prediction_union.find(neighbor));
            const uint32_t killed_target = target_union.link(
                target_union.find(vertex), target_union.find(neighbor));
            if (comparison[killed_comparison] <= comparison[vertex]) {
                return;
            }
            const auto prediction_pair =
                result.prediction.pair_by_birth.find(killed_prediction);
            const auto target_pair = result.target.pair_by_birth.find(killed_target);
            if (prediction_pair != result.prediction.pair_by_birth.end() &&
                target_pair != result.target.pair_by_birth.end()) {
                result.matched_prediction.push_back(prediction_pair->second);
                result.matched_target.push_back(target_pair->second);
            }
        });
    }

    result.matched_prediction.push_back(
        static_cast<int64_t>(result.prediction.pairs.size() - 1));
    result.matched_target.push_back(
        static_cast<int64_t>(result.target.pairs.size() - 1));
    std::vector<uint8_t> prediction_is_matched(result.prediction.pairs.size(), uint8_t{0});
    std::vector<uint8_t> target_is_matched(result.target.pairs.size(), uint8_t{0});
    for (const int64_t index : result.matched_prediction) {
        prediction_is_matched.at(static_cast<size_t>(index)) = uint8_t{1};
    }
    for (const int64_t index : result.matched_target) {
        target_is_matched.at(static_cast<size_t>(index)) = uint8_t{1};
    }
    for (size_t index = 0; index < prediction_is_matched.size(); ++index) {
        if (!prediction_is_matched[index]) {
            result.unmatched_prediction.push_back(static_cast<int64_t>(index));
        }
    }
    for (size_t index = 0; index < target_is_matched.size(); ++index) {
        if (!target_is_matched[index]) {
            result.unmatched_target.push_back(static_cast<int64_t>(index));
        }
    }
    const auto total_end = Clock::now();
    if (profile != nullptr) {
        profile->prediction_pairs_seconds =
            std::chrono::duration<double>(prediction_end - prediction_start).count();
        profile->target_pairs_seconds =
            std::chrono::duration<double>(target_end - prediction_end).count();
        profile->comparison_matching_seconds =
            std::chrono::duration<double>(total_end - target_end).count();
        profile->total_seconds =
            std::chrono::duration<double>(total_end - total_start).count();
    }
    return result;
}

template <class T>
py::array_t<T> vector_to_array(const std::vector<T>& values) {
    py::array_t<T> array(values.size());
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return array;
}

py::dict pair_dict(const PairResult& result) {
    std::vector<uint32_t> births;
    std::vector<int64_t> deaths;
    births.reserve(result.pairs.size());
    deaths.reserve(result.pairs.size());
    py::array_t<bool> essential(result.pairs.size());
    bool* essential_values = essential.mutable_data();
    for (size_t index = 0; index < result.pairs.size(); ++index) {
        births.push_back(result.pairs[index].birth_flat);
        deaths.push_back(result.pairs[index].death_flat);
        essential_values[index] = result.pairs[index].essential;
    }
    py::dict output;
    output["birth_flat"] = vector_to_array(births);
    output["death_flat"] = vector_to_array(deaths);
    output["essential"] = std::move(essential);
    return output;
}

PairResult validated_pair_result(
    const py::array& birth_array,
    const py::array& death_array,
    const py::array& essential_array,
    uint32_t field_size) {
    if (!py::isinstance<py::array_t<uint32_t>>(birth_array)) {
        throw py::type_error("target_birth_flat must have dtype uint32");
    }
    if (!py::isinstance<py::array_t<int64_t>>(death_array)) {
        throw py::type_error("target_death_flat must have dtype int64");
    }
    if (!py::isinstance<py::array_t<bool>>(essential_array)) {
        throw py::type_error("target_essential must have dtype bool");
    }
    for (const auto* array : {&birth_array, &death_array, &essential_array}) {
        if ((array->flags() & py::array::c_style) == 0) {
            throw py::value_error("precomputed target pair arrays must be C-contiguous");
        }
    }
    const py::buffer_info birth_info = birth_array.request();
    const py::buffer_info death_info = death_array.request();
    const py::buffer_info essential_info = essential_array.request();
    if (birth_info.ndim != 1 || death_info.ndim != 1 || essential_info.ndim != 1) {
        throw py::value_error("precomputed target pair arrays must be one-dimensional");
    }
    if (birth_info.size <= 0 || birth_info.size != death_info.size ||
        birth_info.size != essential_info.size) {
        throw py::value_error("precomputed target pair arrays must have equal nonzero lengths");
    }

    const auto* births = static_cast<const uint32_t*>(birth_info.ptr);
    const auto* deaths = static_cast<const int64_t*>(death_info.ptr);
    const auto* essential = static_cast<const bool*>(essential_info.ptr);
    PairResult result;
    result.pairs.reserve(static_cast<size_t>(birth_info.size));
    std::unordered_set<uint32_t> unique_births;
    size_t essential_count = 0;
    for (py::ssize_t index = 0; index < birth_info.size; ++index) {
        if (births[index] >= field_size || deaths[index] >= field_size ||
            deaths[index] < -1) {
            throw py::value_error("precomputed target pair index is outside the target shape");
        }
        if (!unique_births.emplace(births[index]).second) {
            throw py::value_error("precomputed target births must be unique");
        }
        if (essential[index]) {
            ++essential_count;
            if (deaths[index] != -1 || index + 1 != birth_info.size) {
                throw py::value_error(
                    "precomputed target essential pair must be last with death -1");
            }
        } else if (deaths[index] < 0) {
            throw py::value_error("precomputed finite target pairs need a death index");
        }
        const int64_t pair_index = static_cast<int64_t>(result.pairs.size());
        result.pairs.push_back({births[index], deaths[index], essential[index]});
        result.pair_by_birth.emplace(births[index], pair_index);
    }
    if (essential_count != 1) {
        throw py::value_error("precomputed target pairs need exactly one essential branch");
    }
    return result;
}

void add_pairs_with_prefix(py::dict& output, const PairResult& result, const char* prefix) {
    const py::dict pairs = pair_dict(result);
    const py::str birth_key(std::string(prefix) + "_birth_flat");
    const py::str death_key(std::string(prefix) + "_death_flat");
    const py::str essential_key(std::string(prefix) + "_essential");
    output[birth_key] = pairs["birth_flat"];
    output[death_key] = pairs["death_flat"];
    output[essential_key] = pairs["essential"];
}

py::dict matching_dict(const MatchingResult& result) {
    py::dict output;
    add_pairs_with_prefix(output, result.prediction, "prediction");
    add_pairs_with_prefix(output, result.target, "target");
    output["matched_prediction"] = vector_to_array(result.matched_prediction);
    output["matched_target"] = vector_to_array(result.matched_target);
    output["unmatched_prediction"] = vector_to_array(result.unmatched_prediction);
    output["unmatched_target"] = vector_to_array(result.unmatched_target);
    return output;
}

py::dict persistence_pairs_py(const py::array& field_array, float floor) {
    if (!std::isfinite(floor)) {
        throw py::value_error("floor must be finite");
    }
    const FieldView field = validate_field(field_array, "field");
    PairResult result;
    {
        py::gil_scoped_release release;
        result = compute_pairs(field);
    }
    return pair_dict(result);
}

py::dict descending_order_profiled_py(const py::array& field_array) {
    const FieldView field = validate_field(field_array, "field");
    const std::vector<float> values = copy_values(field);
    std::vector<uint32_t> radix_order;
    std::vector<uint32_t> reference_order;
    double radix_seconds;
    double reference_seconds;
    {
        py::gil_scoped_release release;
        const auto radix_start = std::chrono::steady_clock::now();
        radix_order = descending_order_radix(values);
        const auto radix_end = std::chrono::steady_clock::now();
        reference_order = descending_order_reference(values);
        const auto reference_end = std::chrono::steady_clock::now();
        radix_seconds =
            std::chrono::duration<double>(radix_end - radix_start).count();
        reference_seconds =
            std::chrono::duration<double>(reference_end - radix_end).count();
    }
    py::dict output;
    output["radix_order"] = vector_to_array(radix_order);
    output["reference_order"] = vector_to_array(reference_order);
    output["radix_seconds"] = radix_seconds;
    output["reference_seconds"] = reference_seconds;
    return output;
}

py::array_t<uint32_t> neighbor_order_py(
    const py::array& field_array, uint32_t vertex) {
    const FieldView field = validate_field(field_array, "field");
    if (vertex >= field.size) {
        throw py::value_error("vertex is outside the field shape");
    }
    std::vector<uint32_t> neighbors;
    neighbors.reserve(26);
    for_each_neighbor(vertex, field, [&](uint32_t neighbor) {
        neighbors.push_back(neighbor);
    });
    return vector_to_array(neighbors);
}

py::dict induced_matching_py(
    const py::array& prediction_array,
    const py::array& target_array,
    float floor) {
    if (!std::isfinite(floor)) {
        throw py::value_error("floor must be finite");
    }
    const FieldView prediction = validate_field(prediction_array, "prediction");
    const FieldView target = validate_field(target_array, "target");
    if (prediction.depth != target.depth || prediction.height != target.height ||
        prediction.width != target.width) {
        throw py::value_error("prediction and target must have equal shapes");
    }
    MatchingResult result;
    {
        py::gil_scoped_release release;
        result = compute_induced_matching(prediction, target);
    }
    return matching_dict(result);
}

py::dict induced_matching_profiled_py(
    const py::array& prediction_array,
    const py::array& target_array,
    float floor) {
    if (!std::isfinite(floor)) {
        throw py::value_error("floor must be finite");
    }
    const FieldView prediction = validate_field(prediction_array, "prediction");
    const FieldView target = validate_field(target_array, "target");
    if (prediction.depth != target.depth || prediction.height != target.height ||
        prediction.width != target.width) {
        throw py::value_error("prediction and target must have equal shapes");
    }
    MatchingResult result;
    MatchingProfile profile{};
    {
        py::gil_scoped_release release;
        result = compute_induced_matching(prediction, target, &profile);
    }
    py::dict output = matching_dict(result);
    py::dict profile_seconds;
    profile_seconds["prediction_pairs"] = profile.prediction_pairs_seconds;
    profile_seconds["target_pairs"] = profile.target_pairs_seconds;
    profile_seconds["comparison_matching"] = profile.comparison_matching_seconds;
    profile_seconds["total"] = profile.total_seconds;
    output["profile_seconds"] = std::move(profile_seconds);
    return output;
}

py::dict induced_matching_precomputed_target_py(
    const py::array& prediction_array,
    const py::array& target_array,
    const py::array& target_birth_array,
    const py::array& target_death_array,
    const py::array& target_essential_array,
    float floor) {
    if (!std::isfinite(floor)) {
        throw py::value_error("floor must be finite");
    }
    const FieldView prediction = validate_field(prediction_array, "prediction");
    const FieldView target = validate_field(target_array, "target");
    if (prediction.depth != target.depth || prediction.height != target.height ||
        prediction.width != target.width) {
        throw py::value_error("prediction and target must have equal shapes");
    }
    const PairResult target_pairs = validated_pair_result(
        target_birth_array,
        target_death_array,
        target_essential_array,
        target.size);
    MatchingResult result;
    {
        py::gil_scoped_release release;
        result = compute_induced_matching(prediction, target, nullptr, &target_pairs);
    }
    return matching_dict(result);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "persistence_pairs", &persistence_pairs_py,
        py::arg("field"), py::arg("floor"));
    module.def(
        "descending_order_profiled", &descending_order_profiled_py,
        py::arg("field"));
    module.def(
        "neighbor_order", &neighbor_order_py,
        py::arg("field"), py::arg("vertex"));
    module.def(
        "induced_matching", &induced_matching_py,
        py::arg("prediction"), py::arg("target"), py::arg("floor"));
    module.def(
        "induced_matching_profiled", &induced_matching_profiled_py,
        py::arg("prediction"), py::arg("target"), py::arg("floor"));
    module.def(
        "induced_matching_precomputed_target",
        &induced_matching_precomputed_target_py,
        py::arg("prediction"),
        py::arg("target"),
        py::arg("target_birth_flat"),
        py::arg("target_death_flat"),
        py::arg("target_essential"),
        py::arg("floor"));
}
