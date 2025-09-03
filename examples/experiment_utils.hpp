#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/Utils.h>
#include <VarPro/Symbol.h>
#include <VarPro/PyfgTextParser.h>

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
 * @param leaf_dirs the vector to store the leaf directories
 */
void getExperimentDirs(const fs::path &path, std::vector<fs::path> &leaf_dirs)
{
    if (!fs::is_directory(path))
    {
        throw std::invalid_argument("Path " + path.string() + " is not a directory");
    }

    bool has_sub_directory = false;
    for (const auto &entry : fs::directory_iterator(path))
    {
        if (entry.is_directory())
        {
            has_sub_directory = true;
            find_leaf_directories_recursive(entry.path(), leaf_dirs);
        }
    }

    if (!has_sub_directory)
    {
        leaf_dirs.push_back(path);
    }
}

void writeInitializationFile(const fs::path &init_fpath,
                             const VarPro::Problem &problem,
                             const VarPro::Matrix &Y_init)
{
    std::ofstream init_file(init_fpath);
    if (!init_file.is_open())
    {
        throw std::runtime_error("Could not open file " + init_fpath.string() +
                                 " for writing");
    }

    // write each pose to a line
    auto pose_symbol_idxs = problem.getPoseSymbolMap();
    for (const auto &pose_symbol_idx : pose_symbol_idxs)
    {
        VarPro::Symbol pose_symbol = pose_symbol_idx.first;
        VarPro::Index rot_idx = problem.getRotationIdx(pose_symbol);
        VarPro::Index tran_idx = problem.getTranslationIdx(pose_symbol);

        // get the rotation matrix
        // VarPro::Matrix R = problem.


    }

    // write each point to a line

    // write each bearing vector to a line


}
