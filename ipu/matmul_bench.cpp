// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "cxxopts.hpp"

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <popsparse/MatMul.hpp>
#include <popsparse/MatMulParams.hpp>
#include <popsparse/SparsePartitioner.hpp>
#include <popsparse/codelets.hpp>
#include <popsparse/experimental/BlockSparseMatMul.hpp>
#include <poputil/TileMapping.hpp>

namespace {

using EType = float;

///////////////////////////////////////////////////////////////////////////////
// Utilities

poplar::Device attach(size_t count) {
    const auto manager = poplar::DeviceManager::createDeviceManager();
    for (auto&& device : manager.getDevices(poplar::TargetType::IPU, count)) {
        if (device.attach()) {
            return std::move(device);
        }
    }
    throw std::runtime_error("Couldn't attach to device");
}

///////////////////////////////////////////////////////////////////////////////
// Problem setup

struct SparseParams {
    // Sparse LHS, dense RHS
    unsigned groups;
    unsigned batchSize;
    unsigned inputFeatures;
    unsigned outputFeatures;
    unsigned blockSize;
    float density;
    poplar::Type dataType;
    poplar::Type partialsType;
    float availableMemoryProportion;
};

std::ostream& operator<<(std::ostream& out, const SparseParams& p) {
    return out << "SparseParams{batchSize=" << p.batchSize
               << ", inputFeatures=" << p.inputFeatures
               << ", outputFeatures=" << p.outputFeatures
               << ", groups=" << p.groups
               << ", blockSize=" << p.blockSize << ", density=" << p.density
               << ", dataType=" << p.dataType << ", partialsType=" << p.partialsType
               << ", availableMemoryProportion=" << p.availableMemoryProportion << "}";
}

struct SparseProblem {
    SparseParams params;
    // Sparse LHS, block-COO format
    std::vector<float> weights;
    std::vector<unsigned> blockRows;
    std::vector<unsigned> blockColumns;
    // Dense RHS
    std::vector<float> inputs;
};

SparseProblem createProblem(const SparseParams& params, unsigned seed) {
    std::default_random_engine generator(seed);
    SparseProblem result{params, {}, {}, {}, {}};

    if (params.inputFeatures % params.blockSize != 0) {
        std::ostringstream msg;
        msg << "SparseParams inputFeatures (" << params.inputFeatures
            << ") is not divisible by blockSize (" << params.blockSize << ")";
        throw std::invalid_argument(msg.str());
    }
    if (params.outputFeatures % params.blockSize != 0) {
        std::ostringstream msg;
        msg << "SparseParams outputFeatures (" << params.outputFeatures
            << ") is not divisible by blockSize (" << params.blockSize << ")";
        throw std::invalid_argument(msg.str());
    }

    // Create the sparsity pattern

    auto rowBlocks = params.outputFeatures / params.blockSize;
    auto columnBlocks = params.inputFeatures / params.blockSize;
    auto denseBlocks = rowBlocks * columnBlocks;
    auto nonzeroBlocks = static_cast<unsigned>(std::ceil(params.density * denseBlocks));
    std::vector<unsigned> nonzeroRowColumn(denseBlocks);
    std::iota(nonzeroRowColumn.begin(), nonzeroRowColumn.end(), 0u);
    std::shuffle(nonzeroRowColumn.begin(), nonzeroRowColumn.end(), generator);

    for (auto i = 0u; i < nonzeroBlocks; ++i) {
        result.blockRows.push_back(nonzeroRowColumn[i] / columnBlocks);
        result.blockColumns.push_back(nonzeroRowColumn[i] % columnBlocks);
    }

    // Create inputs and weights

    result.inputs.resize(params.batchSize * params.inputFeatures);
    std::generate(result.inputs.begin(), result.inputs.end(), [&generator] {
        return std::uniform_real_distribution<float>(-std::sqrt(3), std::sqrt(3))(generator);
    });

    result.weights.resize(nonzeroBlocks * params.blockSize * params.blockSize);
    auto weightsRange = std::sqrt(3.0f / (params.density * params.inputFeatures));
    std::generate(result.weights.begin(), result.weights.end(), [&generator, weightsRange] {
        return std::uniform_real_distribution<float>(-weightsRange, weightsRange)(generator);
    });

#ifdef DEBUG
    std::cerr << "\nWeights - " << result.weights.size() << "\n";
    for (auto f : result.weights) {
        std::cerr << f << ", ";
    }

    std::cerr << "\nInputs - " << result.inputs.size() << "\n";
    for (auto f : result.inputs) {
        std::cerr << f << ", ";
    }

    std::cerr << "\nBlockRows - " << result.blockRows.size() << "\n";
    for (auto f : result.blockRows) {
        std::cerr << f << ", ";
    }

    std::cerr << "\nBlockColumns - " << result.blockColumns.size() << "\n";
    for (auto f : result.blockColumns) {
        std::cerr << f << ", ";
    }
#endif

    return result;
}

std::vector<float> expectedOutputs(const SparseProblem& problem) {
    auto& p = problem.params;
    auto blockElements = p.blockSize * p.blockSize;
    std::vector<float> result(p.batchSize * p.outputFeatures);

    for (auto block = 0u; block < problem.blockRows.size(); ++block) {
        for (auto i = 0u; i < blockElements; ++i) {
            auto weight = problem.weights[blockElements * block + i];
            auto inputIndex = p.blockSize * problem.blockColumns[block] + (i % p.blockSize);
            auto outputIndex = p.blockSize * problem.blockRows[block] + (i / p.blockSize);
            for (auto j = 0u; j < p.batchSize; ++j) {
                result[p.batchSize * outputIndex + j] +=
                    weight * problem.inputs[p.batchSize * inputIndex + j];
            }
        }
    }

    return result;
}

float meanError(const std::vector<float>& expected, const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        std::ostringstream msg;
        msg << "meanError() size mismatch - expected (" << expected.size() << ") vs actual ("
            << actual.size() << ")";
        throw std::invalid_argument(msg.str());
    }
    auto totalError = 0.0f;
    for (auto i = 0u; i < expected.size(); ++i) {
        totalError += std::abs(expected[i] - actual[i]);
    }
    return totalError / expected.size();
}

///////////////////////////////////////////////////////////////////////////////
// Implementation

enum class ImplementationType { Dense, StaticBlockSparse, DynamicBlockSparse };

ImplementationType parseImplementationType(const std::string& s) {
    if (s == "dense") {
        return ImplementationType::Dense;
    } else if (s == "static-block-sparse") {
        return ImplementationType::StaticBlockSparse;
    } else if (s == "dynamic-block-sparse") {
        return ImplementationType::DynamicBlockSparse;
    } else {
        std::ostringstream msg;
        msg << "Unknown implementation type '" << s << "'";
        throw std::invalid_argument(msg.str());
    }
}

struct Impl {
    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    virtual ~Impl() = default;

    virtual poplar::program::Sequence buildProgram(poplar::Graph&) = 0;
    virtual std::vector<float> run(poplar::Engine&) = 0;
};

void hostWriteFloat(poplar::Graph& graph,
                    const std::string& name,
                    const poplar::Tensor& tensor,
                    poplar::program::Sequence& program) {
    poplar::DebugContext di("hostWriteFloat");
    auto hostWrite = tensor;
    if (tensor.elementType() == poplar::HALF) {
        hostWrite = graph.clone(poplar::FLOAT, tensor);
        popops::castWithOutput(graph, hostWrite, tensor, program, {di, name});
    }
    graph.createHostWrite(name, hostWrite);
}

void hostReadFloat(poplar::Graph& graph,
                   const std::string& name,
                   poplar::Tensor tensor,
                   poplar::program::Sequence& program) {
    poplar::DebugContext di("hostReadFloat");
    if (tensor.elementType() == poplar::HALF) {
        tensor = popops::cast(graph, tensor, poplar::FLOAT, program, {di, name});
    }
    graph.createHostRead(name, tensor);
}

/**
 * Convert the weights tensor to dense, and run a regular dense matmul.
 */
struct DenseImpl : Impl {
    SparseProblem problem;

    explicit DenseImpl(const SparseProblem& problem) : problem(problem) {}

    static std::vector<float> getDenseWeights(const SparseProblem& problem) {
        auto& p = problem.params;
        auto blockElements = p.blockSize * p.blockSize;
        std::vector<float> result(p.inputFeatures * p.outputFeatures);

        for (auto block = 0u; block < problem.blockColumns.size(); ++block) {
            auto startColumn = p.blockSize * problem.blockColumns[block];
            auto startRow = p.blockSize * problem.blockRows[block];
            for (auto i = 0u; i < blockElements; ++i) {
                auto column = startColumn + (i % p.blockSize);
                auto row = startRow + (i / p.blockSize);
                result[p.inputFeatures * row + column] = problem.weights[blockElements * block + i];
            }
        }

        return result;
    }

    poplar::program::Sequence buildProgram(poplar::Graph& graph) {
        auto& p = problem.params;
        poplar::program::Sequence program;
        poplar::DebugContext di("dense");

        poplin::matmul::PlanningCache cache;
        poplar::OptionFlags options({{"partialsType", p.partialsType.toString()}});
        auto lhs = poplin::createMatMulInputLHS(graph, p.dataType, {p.outputFeatures, p.inputFeatures},
                                                {p.inputFeatures, p.batchSize}, {di, "lhsWeights"},
                                                options, &cache);
        hostWriteFloat(graph, "lhsWeights", lhs, program);

        auto rhs = poplin::createMatMulInputRHS(graph, p.dataType, {p.outputFeatures, p.inputFeatures},
                                                {p.inputFeatures, p.batchSize}, {di, "rhsInputs"},
                                                options, &cache);
        hostWriteFloat(graph, "rhsInputs", rhs, program);

        auto result = poplin::matMul(graph, lhs, rhs, program, {di, "matmul"}, options, &cache);
        hostReadFloat(graph, "result", result, program);

        return program;
    }

    std::vector<float> run(poplar::Engine& engine) {
        auto denseWeights = getDenseWeights(problem);
        engine.writeTensor("lhsWeights", denseWeights.data(),
                           denseWeights.data() + denseWeights.size());
        engine.writeTensor("rhsInputs", problem.inputs.data(),
                           problem.inputs.data() + problem.inputs.size());
        engine.run();
        std::vector<float> result(problem.params.outputFeatures * problem.params.batchSize);
        engine.readTensor("result", result.data(), result.data() + result.size());
        return result;
    }
};

struct StaticBlockSparseImpl : Impl {
    SparseProblem problem;
    std::unique_ptr<popsparse::static_::Partitioner<EType>> partitioner;
    popsparse::static_::PlanningCache cache;
    popsparse::CSRMatrix<EType> csrMatrix;

    explicit StaticBlockSparseImpl(const SparseProblem& problem) : problem(problem) {};

    static popsparse::CSRMatrix<EType> getCsrWeights(const SparseProblem& problem) {
        const auto m = problem.params.outputFeatures;
        const auto k = problem.params.inputFeatures;
        const auto blockSize = problem.params.blockSize;
        const auto blockValues = blockSize * blockSize;
        const auto numNZValues = problem.weights.size();
        const auto numNZBlocks = problem.weights.size()/blockValues;
        const auto numRowBlocks = m/blockSize;

        std::vector<size_t> COOColumnIndices(problem.blockColumns.size());
        std::transform(problem.blockColumns.begin(), problem.blockColumns.end(),
                       COOColumnIndices.begin(), [blockSize](auto idx) { return blockSize * idx; });

        std::vector<size_t> COORowIndices(problem.blockRows.size());
        std::transform(problem.blockRows.begin(), problem.blockRows.end(), COORowIndices.begin(),
                       [blockSize](auto idx) { return blockSize * idx; });

        std::vector<std::size_t> rowIndices(numRowBlocks + 1);
        for (const auto &row : COORowIndices) {
            const auto rowBlock = row / blockSize;
            rowIndices.at(rowBlock + 1) += blockValues;
        }
        std::partial_sum(std::next(rowIndices.begin()), rowIndices.end(),
                   std::next(rowIndices.begin()));

        std::vector<std::size_t> perRowBlockIndex(numRowBlocks);
        std::vector<std::size_t> columnIndices(numNZBlocks);
        std::vector<EType> nzValues(numNZValues);

        for (std::size_t block = 0; block < numNZBlocks; ++block) {
            const auto rowBlock = COORowIndices[block] / blockSize;
            const auto dstIndex =
                rowIndices[rowBlock] / blockValues + perRowBlockIndex[rowBlock];
            columnIndices[dstIndex] = COOColumnIndices[block];
            std::copy(problem.weights.begin() + block * blockValues,
                    problem.weights.begin() + (block + 1) * blockValues,
                    nzValues.begin() + dstIndex * blockValues);
            ++perRowBlockIndex[rowBlock];
        }

        auto csrMatrix = popsparse::CSRMatrix<EType>(m, k, nzValues, columnIndices,
            rowIndices, {blockSize, blockSize});

        return csrMatrix;
    }

    poplar::program::Sequence buildProgram(poplar::Graph& graph) override {
        auto& p = problem.params;
        poplar::program::Sequence program;
        poplar::DebugContext di("staticBlockSparse");
        poplar::OptionFlags options{};

        auto matmulParams = popsparse::static_::MatMulParams::createForSparseDense(p.groups,
            p.outputFeatures, p.inputFeatures, p.batchSize);

        partitioner.reset(new popsparse::static_::Partitioner<EType>(matmulParams, p.dataType,
            graph.getTarget(), options, &cache));

        csrMatrix = getCsrWeights(problem);
        auto sparse = popsparse::static_::createSparseDenseMatMulLHS(graph, p.dataType, matmulParams,
                        csrMatrix, {di, "lhsWeights"}, options, &cache);
        hostWriteFloat(graph, "lhsWeights_nzValues", sparse.getNzValuesTensor(), program);

        auto dense = popsparse::static_::createSparseDenseMatMulRHS(graph, p.dataType, matmulParams,
                        csrMatrix, {di, "rhsInputs"}, options, &cache);
        hostWriteFloat(graph, "rhsInputs", dense, program);

        auto result = popsparse::static_::sparseDenseMatMul(graph, sparse, dense, program, false, false,
                                       {di, "matmul"}, options, &cache);
        hostReadFloat(graph, "result", result, program);

        return program;
    }

    std::vector<float> run(poplar::Engine& engine) override {
        auto weights = partitioner->createSparsityDataImpl(csrMatrix);

        engine.writeTensor("lhsWeights_nzValues", weights.nzValues.data(),
                           weights.nzValues.data() + weights.nzValues.size());
        engine.writeTensor("rhsInputs", problem.inputs.data(),
                           problem.inputs.data() + problem.inputs.size());
        engine.run();
        std::vector<float> result(problem.params.outputFeatures * problem.params.batchSize);
        engine.readTensor("result", result.data(), result.data() + result.size());

        return result;
    }

};


struct DynamicBlockSparseImpl : Impl {
    SparseProblem problem;
    std::unique_ptr<popsparse::dynamic::Partitioner<EType>> partitioner;
    popsparse::dynamic::PlanningCache cache;

    explicit DynamicBlockSparseImpl(const SparseProblem& problem) : problem(problem) {}

    static popsparse::COOMatrix<EType> getCooWeights(const SparseProblem& problem) {
        const auto m = problem.params.outputFeatures;
        const auto k = problem.params.inputFeatures;
        const auto blockSize = problem.params.blockSize;

        // Convert block-COO indices to element indices instead of block indices
        std::vector<size_t> columnIndices(problem.blockColumns.size());
        std::transform(problem.blockColumns.begin(), problem.blockColumns.end(),
                       columnIndices.begin(), [blockSize](auto idx) { return blockSize * idx; });

        std::vector<size_t> rowIndices(problem.blockRows.size());
        std::transform(problem.blockRows.begin(), problem.blockRows.end(), rowIndices.begin(),
                       [blockSize](auto idx) { return blockSize * idx; });

        return popsparse::COOMatrix<EType>(m, k, problem.weights, columnIndices, rowIndices,
                                           {blockSize, blockSize});
    }

    poplar::program::Sequence buildProgram(poplar::Graph& graph) override {
        auto& p = problem.params;
        poplar::program::Sequence program;
        poplar::DebugContext di("dynamicBlockSparse");

        auto sparsityParams = popsparse::dynamic::SparsityParams(
            p.blockSize == 1 ? popsparse::dynamic::SparsityType::Element
                             : popsparse::dynamic::SparsityType::Block,
            popsparse::dynamic::SparsityStructure::Unstructured, {p.blockSize, p.blockSize});

        auto params = popsparse::dynamic::MatMulParams::createWithNumNonZeroValues(
            sparsityParams, problem.weights.size(), p.groups, p.outputFeatures, p.inputFeatures,
            p.batchSize);

        poplar::OptionFlags options{};

        partitioner.reset(new popsparse::dynamic::Partitioner<float>(
            params, p.dataType, graph.getTarget(), options, &cache));

        auto lhs = popsparse::dynamic::createSparseDenseMatMulLHS(
            graph, p.dataType, params, {di, "lhsWeights"}, options, &cache);
        hostWriteFloat(graph, "lhsWeights_nzValues", lhs.getNzValuesTensor(), program);
        graph.createHostWrite("lhsWeights_metaInfo", lhs.getMetaInfoTensor());

        auto rhs = popsparse::dynamic::createSparseDenseMatMulRHS(
            graph, p.dataType, params, {di, "rhsInputs"}, options, &cache);
        hostWriteFloat(graph, "rhsInputs", rhs, program);

        auto result = popsparse::dynamic::sparseDenseMatMul(
            graph, lhs, rhs, program, /*transposeLHS=*/false, /*transposeRHS=*/false,
            {di, "matmul"}, options, &cache);
        hostReadFloat(graph, "result", result, program);

        //read back tensors
        auto metaInfo = popops::cast(graph, lhs.getMetaInfoTensor(), lhs.getMetaInfoTensor().elementType(), program);
        graph.createHostRead("readBackMetaInfo", metaInfo);

        return program;
    }

    std::vector<float> run(poplar::Engine& engine) override {
        auto cooWeights = getCooWeights(problem);
        auto weights = partitioner->createSparsityDataImpl(cooWeights);

        std::vector<unsigned short>metaInfoShort(weights.metaInfo.size());
        std::transform(weights.metaInfo.begin(), weights.metaInfo.end(), metaInfoShort.begin(),
                       [](auto i){ return i; });

        engine.writeTensor("lhsWeights_nzValues", weights.nzValues.data(),
                           weights.nzValues.data() + weights.nzValues.size());
        engine.writeTensor("lhsWeights_metaInfo", metaInfoShort.data(),
                           metaInfoShort.data() + metaInfoShort.size());
        engine.writeTensor("rhsInputs", problem.inputs.data(),
                           problem.inputs.data() + problem.inputs.size());
        engine.run();
        std::vector<float> result(problem.params.outputFeatures * problem.params.batchSize);
        engine.readTensor("result", result.data(), result.data() + result.size());

        return result;
    }
};


}  // namespace

namespace std{
    std::istream &operator>>(std::istream &in, poplar::Type &type) {
    std::string token;
    in >> token;
    if (token == "quarter")
        type = poplar::QUARTER;
    else if (token == "half" || token == "float16")
        type = poplar::HALF;
    else if (token == "float" || token == "float32")
        type = poplar::FLOAT;
    else if (token == "unsigned" || token == "uint")
        type = poplar::UNSIGNED_INT;
    else if (token == "int")
        type = poplar::INT;
    else if (token == "ushort")
        type = poplar::UNSIGNED_SHORT;
    else if (token == "short")
        type = poplar::SHORT;
    else if (token == "bool")
        type = poplar::BOOL;
    else if (token == "char")
        type = poplar::CHAR;
    else if (token == "schar")
        type = poplar::SIGNED_CHAR;
    else if (token == "uchar")
        type = poplar::UNSIGNED_CHAR;
    else if (token == "ulonglong")
        type = poplar::UNSIGNED_LONGLONG;
    else if (token == "longlong")
        type = poplar::LONGLONG;
    else
        throw poputil::poplibs_error(
            "Invalid data-type <" + token +
            ">; must be quarter, half (float16), float (float32), uint (unsigned),"
            " int, ushort, short, char, schar, uchar, ulonglong, longlong or bool");
    return in;
    }
}

const poplar::OptionFlags defaultEngineOptions;

int main(int argc, char** argv) {
    SparseParams params{};

    // Argument parsing
    cxxopts::Options options("matmul_bench", "A simple wrapper for matmul ops");

    // DeviceType deviceType = DeviceType::IpuModel2;

    // clang-format off
    options.add_options()
        ("help", "Produce help message")
        ("compile-only", "Stop after compilation; don't run the program",
	 cxxopts::value<bool>()->default_value("false"))
        ("profile", "Enable profiling and print profiling report",
	 cxxopts::value<bool>()->default_value("false"))
        ("profile-dir",
         "Enable profiling and write profile files to the specified directory.",
	 cxxopts::value<std::string>()->default_value("./profile"))
        ("ignore-data", "Don't validate results")
        ("implementation", "Implementation (dense | static-block-sparse | dynamic-block-sparse)",
	 cxxopts::value<std::string>()->default_value("dense"))
        ("data-type", "Data type of operands", cxxopts::value<poplar::Type>()->default_value("half"))
        ("partials-type", "Data type of partial accumulators", cxxopts::value<poplar::Type>()->default_value("float"))
        ("groups", "Number of groups", cxxopts::value<unsigned>()->default_value("1"))
        ("m, output", "Rows in left-hand operand", cxxopts::value<unsigned>())
        ("k, input", "Columns in left-hand operand/Rows in right-hand operand", cxxopts::value<unsigned>())
        ("n, batch", "Columns in right-hand operand", cxxopts::value<unsigned>()) 
        ("block-size",
         "Block size as rows and columns (only square blocks are supported)",
         cxxopts::value<unsigned>()->default_value("1"))
        ("density", "Proportion of non-zero elements of the sparse matrix",
	 cxxopts::value<float>()->default_value("1.0"))
        ("available-memory-proportion",
         "The estimated proportion of memory available to perform this operation",
         cxxopts::value<float>()->default_value("0.6"))
        ("transpose-lhs",
         "Transpose the left-hand operand of the matmul such that the matmul "
         "becomes {k, m} * {m, n} = {k, n}",
	 cxxopts::value<bool>()->default_value("false"))
        ("transpose-rhs",
         "Transpose the right-hand operand of the matmul",
	 cxxopts::value<bool>()->default_value("false"))
        ("matmul-options", 
         "Options to use for the matrix multiplication, specified as a JSON "
         "string, e.g. {\"key\":\"value\"}",
	 cxxopts::value<std::string>())
    ;
    // clang-format on

    auto result = cxxopts::ParseResult{};
    try {
        result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << "\n";
            return 0;
        }
    } catch (std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    // bool compileOnly = result["compile-only"].as<bool>();;
    bool ignoreData = result["ignore-data"].as<bool>();
    bool profile = result["profile"].as<bool>();
    std::string profileDir = result["profile-dir"].as<std::string>();;
    std::string implArg = result["implementation"].as<std::string>();
    // bool transposeLHS = result["transpose-lhs"].as<bool>();
    // bool transposeRHS = result["transpose-rhs"].as<bool>();
    std::string matmulOptions{};
    if (result.count("matmul-options")) {
        matmulOptions = result["matmul-options"].as<std::string>();
    }

    params.groups = result["groups"].as<unsigned>();
    if (result.count("n")) {
        params.batchSize = result["n"].as<unsigned>();
    } else {
        std::cerr << "-n, --batch required\n";
	return 1;
    }
    if (result.count("m")) {
        params.outputFeatures = result["m"].as<unsigned>();
    } else {
        std::cerr << "-m, --output required\n";
	return 1;
    }
    if (result.count("k")) {
        params.inputFeatures = result["k"].as<unsigned>();
    } else {
        std::cerr << "-k, --intput required\n";
	return 1;
    }
    params.blockSize = result["block-size"].as<unsigned>();
    params.density = result["density"].as<float>();
    params.availableMemoryProportion = result["available-memory-proportion"].as<float>();
    params.dataType = result["data-type"].as<poplar::Type>();
    params.partialsType = result["partials-type"].as<poplar::Type>();
    

    poplar::OptionFlags engineOptions = defaultEngineOptions;
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    engineOptions.set("autoReport.directory", profileDir);

    const std::vector<unsigned> supportedBlockLengths = {1, 4, 8, 16};
    if (std::find(supportedBlockLengths.begin(), supportedBlockLengths.end(),
        params.blockSize) == supportedBlockLengths.end()) {
        throw poputil::poplibs_error("Block size " + std::to_string(params.blockSize)
                                     + " not supported");
    }

    if (params.inputFeatures % params.blockSize) {
        throw poputil::poplibs_error("Input size must be an integer multiple of "
                                    "rows in a block");
    }

    if (params.outputFeatures % params.blockSize) {
        throw poputil::poplibs_error("output size must be an integer multiple of "
                                    "columns in a block");
    }

    std::cout << "Running with parameters: " << params << "\n";

    auto implementation = parseImplementationType(implArg);
    auto problem = createProblem(params, 123u);

    // Compilation
    std::cerr << "\n[matmul_wrapper.cpp] Building..." << std::endl;
    std::unique_ptr<Impl> impl;
    if (implementation == ImplementationType::Dense) {
        impl.reset(new DenseImpl(problem));
    } else if (implementation == ImplementationType::StaticBlockSparse) {
        impl.reset(new StaticBlockSparseImpl(problem));
    } else if (implementation == ImplementationType::DynamicBlockSparse) {
        impl.reset(new DynamicBlockSparseImpl(problem));
    }
    const auto device = attach(1);
    poplar::Graph graph(device.getTarget());
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    popsparse::addCodelets(graph);
    poplar::Engine engine(graph, impl->buildProgram(graph), engineOptions);

    // Execution
    std::cerr << "[matmul_wrapper.cpp] Running..." << std::endl;
    engine.load(device);
    auto actual = impl->run(engine);

    // Check results
    if (!ignoreData) {
        std::cerr << "[matmul_wrapper.cpp] Checking..." << std::endl;
        auto expected = expectedOutputs(problem);
        std::cerr << "[matmul_wrapper.cpp] Final error: " << meanError(expected, actual) << std::endl;
    }

    if (profile) {
        engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
    }

    return 0;
}
