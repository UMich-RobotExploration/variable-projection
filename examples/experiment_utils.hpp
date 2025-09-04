#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/Utils.h>
#include <VarPro/Symbol.h>
#include <VarPro/PyfgTextParser.h>

#include <filesystem>
namespace fs = std::filesystem;

using PoseChain = std::vector<VarPro::Symbol>;
using PoseChains = std::vector<PoseChain>;
using RPM = VarPro::RelativePoseMeasurement;

PoseChains getRobotPoseChains(const VarPro::Problem &problem)
{
    // get all of the unique pose characters
    std::set<unsigned char> seen_pose_chars;
    for (auto const &all_pose_symbols : problem.getPoseSymbolMap())
    {
        VarPro::Symbol pose_symbol = all_pose_symbols.first;
        seen_pose_chars.insert(pose_symbol.chr());
    }

    // get a sorted list of the unique pose characters
    std::vector<unsigned char> unique_pose_chars = {seen_pose_chars.begin(),
                                                    seen_pose_chars.end()};
    std::sort(unique_pose_chars.begin(), unique_pose_chars.end());

    // for each unique pose character, get the pose symbols (sorted)
    PoseChains robot_pose_chains;
    for (auto const &pose_char : unique_pose_chars)
    {
        PoseChain robot_pose_chain = problem.getPoseSymbols(pose_char);
        std::sort(robot_pose_chain.begin(), robot_pose_chain.end());
        robot_pose_chains.push_back(robot_pose_chain);
    }

    // return the robot pose chains
    return robot_pose_chains;
}

void saveSolutions(const VarPro::Problem &problem,
                   const VarPro::Matrix &aligned_soln,
                   const std::string &pyfg_fpath)
{
    /**
     * @brief for each robot, save the solution to a .tum file e.g.
     * data/plaza2.pyfg -> /tmp/plaza2/varpro_0.tum
     * or
     * data/marine_two_robots.pyfg -> /tmp/marine_two_robots/varpro_0.tum,
     * /tmp/marine_two_robots/varpro_1.tum
     *
     */

    // strip the .pyfg extension and the data/ prefix
    size_t data_length = std::string("data/").length();
    size_t pyfg_index = pyfg_fpath.find(".pyfg");
    std::string save_dir_name =
        pyfg_fpath.substr(data_length, pyfg_index - data_length);

    // if save_dir_name starts with /, then remove it
    if (save_dir_name[0] == '/')
    {
        save_dir_name = save_dir_name.substr(1);
    }
    std::string save_dir_path = "/tmp/" + save_dir_name;

    // create the directory if it does not exist. Make sure to recursively create
    // the parent directories
    if (!std::filesystem::exists(save_dir_path))
    {
        std::filesystem::create_directories(save_dir_path);
    }

    std::string save_path = save_dir_path + "/varpro_";

    // get the different robot pose chains
    PoseChains robot_pose_chains = getRobotPoseChains(problem);

    // if tiers.pyfg, then we have four robots
    if (pyfg_fpath == "data/tiers.pyfg" && robot_pose_chains.size() != 4)
    {
        throw std::runtime_error("Expected 4 robots in tiers.pyfg");
    }

    // enumerate over the robot pose chains
    for (size_t robot_index = 0; robot_index < robot_pose_chains.size();
         robot_index++)
    {
        // get the robot pose chain
        PoseChain robot_pose_chain = robot_pose_chains[robot_index];

        // save the estimated poses for this robot
        std::string robot_save_path =
            save_path + std::to_string(robot_index) + ".tum";
        saveSolnToTum(robot_pose_chain, problem, aligned_soln, robot_save_path);
        // std::cout << "Saved " << robot_save_path << std::endl;

        std::string g2o_path = save_path + std::to_string(robot_index) + ".g2o";
        saveSolnToG20(robot_pose_chain, problem, aligned_soln, g2o_path);
        // std::cout << "Saved " << g2o_path << std::endl;
    }
}

fs::path findPyfgInDir(const fs::path &dir_path)
{
    for (const auto &entry : fs::directory_iterator(dir_path))
    {
        if (entry.path().extension() == ".pyfg")
        {
            return entry.path();
        }
    }
    throw std::runtime_error("No .pyfg file found in directory " + dir_path.string());
}

/**
 * @brief Searches for all of the experiment directories under a path (they are
 * the leaf directories)
 *
 * @param path the path to search
 * @param exp_dirs the vector to store the leaf directories
 */
void getExperimentDirsRecursive(const fs::path &path, std::vector<fs::path> &exp_dirs)
{
    if (!fs::is_directory(path))
    {
        throw std::invalid_argument("Path " + path.string() + " is not a directory");
    }

    bool has_pyfg = false;
    for (const auto &entry : fs::directory_iterator(path))
    {
        if (entry.is_directory())
        {
            getExperimentDirsRecursive(entry.path(), exp_dirs);
        }
        if (entry.path().extension() == ".pyfg")
        {
            has_pyfg = true;
        }
    }

    if (has_pyfg)
    {
        exp_dirs.push_back(path);
    }
}

std::uintmax_t dir_size(const fs::path& dir) {
    std::uintmax_t size = 0;
    std::error_code ec;

    if (!fs::exists(dir, ec)) return 0;

    for (auto const& entry : fs::recursive_directory_iterator(
             dir, fs::directory_options::skip_permission_denied, ec))
    {
        if (entry.is_regular_file(ec)) {
            size += entry.file_size(ec);
        }
    }
    return size;
}


void writeInitializationFile(const fs::path &init_fpath,
                             const VarPro::Problem &problem,
                             const VarPro::Matrix &Y_init)
{
    // if the directory does not exist, create it
    if (!std::filesystem::exists(init_fpath.parent_path()))
    {
        std::filesystem::create_directories(init_fpath.parent_path());
    }

    std::ofstream init_file(init_fpath);
    if (!init_file.is_open())
    {
        throw std::runtime_error("Could not open file " + init_fpath.string() +
                                 " for writing");
    }

    // write each pose to a line
    // pose at rank k: VERTEX_POSE <symbol> <p1> ... <pk> <r11> ... <r1k> ... <rd1> ... <rdk>
    // where p is the translation and r is the rotation matrix in row-major order
    for (const auto &pose_symbol_idx : problem.getPoseSymbolMap())
    {
        VarPro::Symbol pose_symbol = pose_symbol_idx.first;
        VarPro::Matrix R = problem.getRotationFromSymbol(Y_init, pose_symbol);

        // check that the rotation is rank x dim
        checkMatrixShape("writeInitializationFile::R", problem.getRelaxationRank(),
                         problem.dim(), R.rows(), R.cols());

        //  should be a column vector
        VarPro::Matrix t = problem.getTranslationFromSymbol(Y_init, pose_symbol);
        checkMatrixShape("writeInitializationFile::t", problem.getRelaxationRank(), 1,
                         t.rows(), t.cols());

        std::string pose_line = "VERTEX_POSE " + pose_symbol.string() + " ";
        for (int i = 0; i < t.rows(); i++)
        {
            pose_line += std::to_string(t(i, 0)) + " ";
        }
        for (int i = 0; i < R.rows(); i++)
        {
            for (int j = 0; j < R.cols(); j++)
            {
                pose_line += std::to_string(R(i, j));
                if (j < R.cols() - 1 || i < R.rows() - 1) // add space if not last element
                {
                    pose_line += " ";
                }
            }
        }
        init_file << pose_line << std::endl;
    }

    // write each point to a line
    // point at rank k: VERTEX_POINT  <symbol> <p1> ... <pk>
    for (const auto &landmark_symbol_idx : problem.getLandmarkSymbolMap())
    {
        VarPro::Symbol landmark_symbol = landmark_symbol_idx.first;
        VarPro::Matrix p = problem.getTranslationFromSymbol(Y_init, landmark_symbol);
        checkMatrixShape("writeInitializationFile::p", problem.getRelaxationRank(), 1,
                         p.rows(), p.cols());

        std::string point_line = "VERTEX_POINT " + landmark_symbol.string() + " ";
        for (int i = 0; i < p.rows(); i++)
        {
            point_line += std::to_string(p(i, 0));
            if (i < p.rows() - 1) // add space if not last element
            {
                point_line += " ";
            }
        }
        init_file << point_line << std::endl;
    }

    // write each bearing vector to a line
    // bearing at rank k: VERTEX_BEARING  <symbol1> <symbol2> <b1> ... <bk>
    for (const auto &range_measurement : problem.getRangeMeasurements())
    {
        VarPro::SymbolPair range_symbol_pair = std::make_pair(range_measurement.first_id,
                                                              range_measurement.second_id);
        VarPro::Matrix b =
            problem.getBearingFromRangeSymbolPair(Y_init, range_symbol_pair);
        checkMatrixShape("writeInitializationFile::b", problem.getRelaxationRank(), 1,
                         b.rows(), b.cols());

        std::string bearing_line = "VERTEX_BEARING " + range_measurement.first_id.string() + " " +
                                   range_measurement.second_id.string() + " ";
        for (int i = 0; i < b.rows(); i++)
        {
            bearing_line += std::to_string(b(i, 0));
            if (i < b.rows() - 1) // add space if not last element
            {
                bearing_line += " ";
            }
        }
        init_file << bearing_line << std::endl;
    }

    init_file.close();
    std::cout << "Wrote initialization to " << init_fpath << std::endl;
}

VarPro::Matrix readInitializationFile(const fs::path &init_fpath,
                                      const VarPro::Problem &problem)
{
    // check if the file exists
    if (!std::filesystem::exists(init_fpath))
    {
        throw std::runtime_error("Initialization file " + init_fpath.string() +
                                 " does not exist");
    }

    std::ifstream init_file(init_fpath);
    if (!init_file.is_open())
    {
        throw std::runtime_error("Could not open initialization file " +
                                 init_fpath.string() + " for reading");
    }

    VarPro::Matrix Y_init = VarPro::Matrix::Zero(problem.getExpectedVariableSize(),
                                                 problem.getRelaxationRank());

    std::string line;
    while (std::getline(init_file, line))
    {
        std::istringstream iss(line);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>{}};

        if (tokens.size() == 0)
        {
            continue;
        }

        if (tokens[0] == "VERTEX_POSE")
        {
            if (tokens.size() < 2 + problem.getRelaxationRank() +
                                    problem.dim() * problem.getRelaxationRank())
            {
                throw std::runtime_error("Invalid VERTEX_POSE line in initialization file " +
                                         init_fpath.string());
            }
            VarPro::Symbol pose_symbol(tokens[1]);
            VarPro::Matrix t(problem.getRelaxationRank(), 1);
            for (int i = 0; i < problem.getRelaxationRank(); i++)
            {
                t(i, 0) = std::stod(tokens[2 + i]);
            }
            VarPro::Matrix R(problem.getRelaxationRank(), problem.dim());
            for (int i = 0; i < problem.getRelaxationRank(); i++)
            {
                for (int j = 0; j < problem.dim(); j++)
                {
                    R(i, j) = std::stod(tokens[2 + problem.getRelaxationRank() + i * problem.dim() + j]);
                }
            }

            // set the rotation
            Index rot_idx = problem.getRotationIdx(pose_symbol);
            Y_init.block(rot_idx * problem.dim(), 0, problem.dim(),
                         problem.getRelaxationRank()) = R.transpose();

            // set the translation
            if (problem.getFormulation() != VarPro::Formulation::Implicit)
            {
                Index tran_idx = problem.getTranslationIdx(pose_symbol);
                Y_init.block(tran_idx, 0, 1, problem.getRelaxationRank()) = t.transpose();
            }
        }

        if (tokens[0] == "VERTEX_POINT" && problem.getFormulation() != VarPro::Formulation::Implicit)
        {
            if (tokens.size() < 2 + problem.getRelaxationRank())
            {
                throw std::runtime_error("Invalid VERTEX_POINT line in initialization file " +
                                         init_fpath.string());
            }
            VarPro::Symbol point_symbol(tokens[1]);
            VarPro::Matrix p(problem.getRelaxationRank(), 1);
            for (int i = 0; i < problem.getRelaxationRank(); i++)
            {
                p(i, 0) = std::stod(tokens[2 + i]);
            }

            // set the point
            Index point_idx = problem.getTranslationIdx(point_symbol);
            Y_init.block(point_idx, 0, 1, problem.getRelaxationRank()) = p.transpose();
        }

        if (tokens[0] == "VERTEX_BEARING")
        {
            if (tokens.size() < 3 + problem.getRelaxationRank())
            {
                throw std::runtime_error("Invalid VERTEX_BEARING line in initialization file " +
                                         init_fpath.string());
            }
            VarPro::Symbol symbol1(tokens[1]);
            VarPro::Symbol symbol2(tokens[2]);
            VarPro::Matrix b(problem.getRelaxationRank(), 1);
            for (int i = 0; i < problem.getRelaxationRank(); i++)
            {
                b(i, 0) = std::stod(tokens[3 + i]);
            }

            // set the bearing
            VarPro::SymbolPair range_symbol_pair = std::make_pair(symbol1, symbol2);
            Index bearing_idx = problem.getRangeIdx(range_symbol_pair);
            Y_init.block(bearing_idx, 0, 1, problem.getRelaxationRank()) = b.transpose();
        }
    }

    init_file.close();
    return Y_init;
}