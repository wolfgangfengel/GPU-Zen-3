#include "common/options.h"

Options::Options() {}

void Options::SetIntOptions(const std::string& name, const int value) {
    int_options_[name] = value;
}

const int Options::GetIntOptions(const std::string& name) const {
    return int_options_.at(name);
}

void Options::SetRealOptions(const std::string& name, const real value) {
    real_options_[name] = value;
}

const real Options::GetRealOptions(const std::string& name) const {
    return real_options_.at(name);
}

void Options::SetVectorOptions(const std::string& name, const VectorX& value) {
    vector_options_[name] = value;
}

const VectorX Options::GetVectorOptions(const std::string& name) const {
    return vector_options_.at(name);
}

void Options::SetVectorIOptions(const std::string& name, const VectorXI& value) {
	vectori_options_[name] = value;
}

const VectorXI Options::GetVectorIOptions(const std::string& name) const {
	return vectori_options_.at(name);
}

void Options::SetMatrixOptions(const std::string& name, const MatrixX& value) {
    matrix_options_[name] = value;
}

const MatrixX Options::GetMatrixOptions(const std::string& name) const {
    return matrix_options_.at(name);
}

void Options::SetBoolOptions(const std::string &name, const bool value) {
    if (value) {
        bool_options_.insert(name);
    } else {
        bool_options_.erase(name);
    }
}

const bool Options::GetBoolOptions(const std::string &name) const {
    return bool_options_.find(name) != bool_options_.end();
}