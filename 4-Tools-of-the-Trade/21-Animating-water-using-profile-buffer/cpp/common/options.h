#ifndef FLUID_TOPO_OPTIONS_H
#define FLUID_TOPO_OPTIONS_H

#include "config.h"

class Options {
public:
    Options();

    void SetIntOptions(const std::string& name, const int value);
    const int GetIntOptions(const std::string& name) const;

    void SetRealOptions(const std::string& name, const real value);
    const real GetRealOptions(const std::string& name) const;

    void SetVectorOptions(const std::string& name, const VectorX& value);
    const VectorX GetVectorOptions(const std::string& name) const;

	void SetVectorIOptions(const std::string& name, const VectorXI& value);
	const VectorXI GetVectorIOptions(const std::string& name) const;

    void SetMatrixOptions(const std::string& name, const MatrixX& value);
    const MatrixX GetMatrixOptions(const std::string& name) const;

    void SetBoolOptions(const std::string& name, const bool value);
    const bool GetBoolOptions(const std::string& name) const;

private:
    std::unordered_map<std::string, int> int_options_;
    std::unordered_map<std::string, real> real_options_;
    std::unordered_map<std::string, VectorX> vector_options_;
    std::unordered_map<std::string, MatrixX> matrix_options_;
    std::unordered_set<std::string> bool_options_;
    std::unordered_map<std::string, VectorXI> vectori_options_;
};

#endif //FLUID_TOPO_OPTIONS_H
