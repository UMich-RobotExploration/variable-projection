/**
 * @file PyfgTextParser.h
 * @author
 * @brief Utilities to parse a text file written in the PyFG format
 * @version 0.1
 * @date 2023-10-23
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/Symbol.h>

#include <fstream>
#include <iostream>
#include <string>

namespace VarPro {

/**
 * @brief Takes a text file written in the PyFG format and parses it into a
 * VarPro::Problem object
 *
 * @param filename the name of the file to parse
 * @return VarPro::Problem the parsed problem
 */
Problem parsePyfgTextToProblem(const std::string &filename);

} // namespace VarPro
