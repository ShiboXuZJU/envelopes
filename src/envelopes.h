/*
This file contains the declarations of the classes and functions used in the envelopes library.

How to add a new class of envelope (e.g. NewEnv):
1. Add a new elemment NEWENV in enum class EnvType;
2. Create a new class NewEnv : public AbstractEnvelope {... }.
    Constructor, realModel, complexModel, serializationState and Constructor from serialized bytes data are necessary;
3. Add a case for NewEnv in the switch-case statement in function "deserialization";
4. Add a bind class for NewEnv in the "bind.cpp" file;
*/

#pragma once

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <variant>
#include <vector>
#include <algorithm>
#include <sstream>
#include <string>
#include <memory>
#include <complex>
#include <type_traits>

#ifndef CERF_AS_CPP
#define CERF_AS_CPP
#endif
#include "libcerf/cerf.h" // copy from vcpkg/installed/x64-windows/include/cerf.h

#define REAL float
#define COMPLEX std::complex<float>

#define UNION std::variant
#define VEC std::vector
#define REAL_VEC VEC<REAL>
#define COMPLEX_VEC VEC<COMPLEX>
#define DICT std::unordered_map
#define SET std::unordered_set
#define TP std::tuple
#define SP std::shared_ptr
#define REAL_VEC_SP SP<REAL_VEC>
#define COMPLEX_VEC_SP SP<COMPLEX_VEC>
#define ANY_VEC_SP UNION<REAL_VEC_SP, COMPLEX_VEC_SP>

#define ROUND std::llround
#define LLINT long long int

// #define SERIALIZATION_VEC_INDEX_TYPE size_t
// SERIALIZATION_VEC_INDEX_TYPE.Generally, as long as the number of individual envelopes is less than 2^32, this should be fine.
#define SVIT uint32_t

#define LOG(x) std::cout << x << std::endl

double M_PI = 3.14159265358979323846; // pi
std::complex<double> I(0, 1);

double TIME_ATOL = 1e-5;
double FREQ_ATOL = 1e-5;
double FLOAT_ATOL = 1e-8;
LLINT END_PADDING = 2;

enum class EnvType : unsigned char
{
    GAUSSIAN,
    GAUSSIANDRAG,
    COSINEDRAG,
    TRIANGLE,
    RECT,
    FLATTOP,
    RIPPLERECT,
    MIXEXP,
    ENVMIX,
    ENVSHIFT,
    ENVPROD,
    ENVSUM
};

struct EnvFeature
{
    SP<VEC<LLINT>> values;
    bool operator==(const EnvFeature &other) const
    {
        return values->size() == other.values->size() && std::equal(values->begin(), values->end(), other.values->begin());
    }
    std::string toString() const
    {
        std::stringstream ss;
        ss << "[";
        for (auto &value : *values)
        {
            ss << value << " ";
        }
        ss << "]";
        return ss.str();
    }
};

namespace std
{
    template <>
    struct hash<EnvFeature>
    {
        size_t operator()(const EnvFeature &tuple) const
        {
            size_t result = hash<LLINT>()((*tuple.values)[0]);
            for (size_t i = 1; i < tuple.values->size(); i++)
            {
                result ^= hash<LLINT>()((*tuple.values)[i]);
            }
            return result;
        }
    };
}

template <typename T>
class LazyArray : public std::enable_shared_from_this<LazyArray<T>>
{
public:
    T scale_;
    SP<VEC<T>> array_;
    VEC<SP<VEC<T>>> arrayBuffers_;
    LazyArray(T scale, SP<VEC<T>> &array) : scale_(scale), array_(array) {}
    size_t size() const { return array_->size(); }
    T operator[](size_t index) const
    {
        T result = (*array_)[index] * scale_;
        for (auto &buffer : arrayBuffers_)
        {
            result *= (*buffer)[index];
        }
        return result;
    }
    SP<VEC<T>> eval() // Time consuming.
    {
        SP<VEC<T>> result = std::make_shared<VEC<T>>(this->size());
        for (size_t i = 0; i < this->size(); i++)
        {
            (*result)[i] = (*this)[i];
        }
        return result;
    }
    SP<LazyArray<T>> operator*=(const REAL &scale)
    {
        scale_ *= scale;
        return shared_from_this();
    }
    SP<LazyArray<COMPLEX>> operator*=(const COMPLEX &scale)
    {
        if constexpr (std::is_same_v<T, COMPLEX>)
        {
            scale_ *= scale;
            return shared_from_this();
        }
        throw std::runtime_error("LazyArray<REAL> cannot be multiplied by COMPLEX.");
    }
    SP<LazyArray<T>> operator*=(SP<LazyArray<REAL>> &other)
    {
        if (this->size() != other->size())
        {
            throw std::runtime_error("LazyArray cannot be multiplied by other LazyArray with different sizes.");
        }
        scale_ *= other->scale_;
        arrayBuffers_.reserve(arrayBuffers_.size() + 1 + other->arrayBuffers_.size());
        arrayBuffers_.push_back(other->array_);
        arrayBuffers_.insert(arrayBuffers_.end(), other->arrayBuffers_.begin(), other->arrayBuffers_.end());
        return shared_from_this();
    }
    SP<LazyArray<COMPLEX>> operator*=(SP<LazyArray<COMPLEX>> &other)
    {
        if constexpr (std::is_same_v<T, COMPLEX>)
        {
            if (this->size() != other->size())
            {
                throw std::runtime_error("LazyArray cannot be multiplied by other LazyArray with different sizes.");
            }
            scale_ *= other->scale_;
            arrayBuffers_.reserve(arrayBuffers_.size() + 1 + other->arrayBuffers_.size());
            arrayBuffers_.push_back(other->array_);
            arrayBuffers_.insert(arrayBuffers_.end(), other->arrayBuffers_.begin(), other->arrayBuffers_.end());
            return shared_from_this();
        }
        throw std::runtime_error("LazyArray<REAL> cannot be multiplied by LazyArray<COMPLEX>.");
    }
};

class WaveCache
{
public:
    double resolution_;
    LLINT resolutionOverAtol_;
    DICT<std::string, DICT<LLINT, SP<DICT<EnvFeature, REAL_VEC_SP>>>> realCache;
    DICT<std::string, DICT<LLINT, SP<DICT<EnvFeature, COMPLEX_VEC_SP>>>> complexCache;
    WaveCache(double resolution) : resolution_(resolution), resolutionOverAtol_(ROUND(resolution / TIME_ATOL)) {}
    double resolution() const { return resolution_; }
    std::string toString() const
    {
        std::stringstream ss;
        ss << "resolution: " << resolution_ << std::endl
           << std::endl;
        ss << "realCache:" << std::endl;
        for (auto &pair1 : realCache)
        {
            ss << "  ";
            ss << pair1.first << ":" << std::endl;
            for (auto &pair2 : pair1.second)
            {
                ss << "    ";
                ss << pair2.first << ":" << std::endl;
                for (auto &pair3 : (*pair2.second))
                {
                    ss << "      ";
                    ss << pair3.first.toString() << std::endl;
                }
            }
        }
        ss << std::endl;
        ss << "complexCache:\n";
        for (auto &pair1 : complexCache)
        {
            ss << "  ";
            ss << pair1.first << ":" << std::endl;
            for (auto &pair2 : pair1.second)
            {
                ss << "    ";
                ss << pair2.first << ":" << std::endl;
                for (auto &pair3 : (*pair2.second))
                {
                    ss << "      ";
                    ss << pair3.first.toString() << std::endl;
                }
            }
        }
        return ss.str();
    }
    /*
    Usually, the same envelopes with different baseStart values have different discrete-time waveforms.
    In this case, we need to store the cache for each baseStart value separately.
    */
    std::tuple<double, double, SP<DICT<EnvFeature, REAL_VEC_SP>>> getRealCache(std::string envName, double start);
    std::tuple<double, double, SP<DICT<EnvFeature, COMPLEX_VEC_SP>>> getComplexCache(std::string envName, double start);
    /*
    However, exceptions exist such as "MixExp".
    These envelopes use the same cache for all baseStart values. In this case, the original "baseStart" key is set to 0.
    */
    SP<DICT<EnvFeature, REAL_VEC_SP>> getRealCache(std::string envName);
    SP<DICT<EnvFeature, COMPLEX_VEC_SP>> getComplexCache(std::string envName);
};

class AbstractEnvelope : public std::enable_shared_from_this<AbstractEnvelope>
{
public:
    EnvType type_;

    virtual ~AbstractEnvelope() {}
    virtual double start() = 0;
    virtual double end() = 0;
    virtual bool is_complex() = 0;
    virtual double duration() { return end() - start(); }
    virtual std::string toString() const { return "AbstractEnvelope"; }
    virtual SP<AbstractEnvelope> operator+(const double &zero);
    virtual SP<AbstractEnvelope> operator+(SP<AbstractEnvelope> &env);
    virtual SP<AbstractEnvelope> operator+=(SP<AbstractEnvelope> &env);
    virtual SP<AbstractEnvelope> operator*(const REAL &scale);
    virtual SP<AbstractEnvelope> operator*(const COMPLEX &scale);
    virtual SP<AbstractEnvelope> operator>>(const double &dt);
    virtual TP<double, SP<LazyArray<REAL>>> realTimeFunc(WaveCache &wc, double dt) = 0;
    virtual TP<double, SP<LazyArray<COMPLEX>>> complexTimeFunc(WaveCache &wc, double dt) = 0;
    virtual TP<std::string *, void *> serializationState() const = 0;
    virtual void setDeserializedEnvPtr(std::string &serialized, size_t &current, VEC<SP<AbstractEnvelope>> &DeserializedEnvs) = 0;

    friend SP<AbstractEnvelope> operator+(const double &zero, SP<AbstractEnvelope> &env);
    friend SP<AbstractEnvelope> operator*(const REAL &scale, SP<AbstractEnvelope> &env);
    friend SP<AbstractEnvelope> operator*(const COMPLEX &scale, SP<AbstractEnvelope> &env);
};

class Envelope : public AbstractEnvelope
{
public:
    std::string name_;
    double start_;
    double end_;
    bool is_complex_;
    REAL amp_;
    EnvFeature feature_;

    Envelope(double start, double end, bool is_complex, REAL amp, std::string name)
        : start_(start), end_(end), is_complex_(is_complex), amp_(amp), name_(name) {}
    double start() override { return start_; }
    double end() override { return end_; }
    bool is_complex() override { return is_complex_; }
    std::string toString() const override { return name_; }
    EnvFeature feature() { return feature_; };
    virtual REAL_VEC_SP realModel(double baseStart, size_t sampleCount, double resolution)
    {
        throw std::runtime_error("NOT implemented.");
    }
    virtual COMPLEX_VEC_SP complexModel(double baseStart, size_t sampleCount, double resolution)
    {
        throw std::runtime_error("NOT implemented.");
    }
    TP<double, SP<LazyArray<REAL>>> realTimeFunc(WaveCache &wc, double dt) override;
    TP<double, SP<LazyArray<COMPLEX>>> complexTimeFunc(WaveCache &wc, double dt) override;
    void setDeserializedEnvPtr(std::string &serialized, size_t &current, VEC<SP<AbstractEnvelope>> &DeserializedEnvs) override {}
};

class Gaussian : public Envelope
{
public:
    double t0_;
    double w_;
    double sigma_;

    Gaussian(double t0, double w, REAL amp)
        : Envelope(t0 - 3 * w, t0 + 3 * w, false, amp, "Gaussian"), t0_(t0), w_(w)
    {
        sigma_ = w_ / sqrt(8.0 * log(2.0));
        feature_.values = std::make_shared<VEC<LLINT>>(1);
        feature_.values->at(0) = (LLINT)ROUND(w_ / TIME_ATOL);
        type_ = EnvType::GAUSSIAN;
    }
    REAL_VEC_SP realModel(double baseStart, size_t sampleCount, double resolution) override
    {
        REAL_VEC_SP wave = std::make_shared<REAL_VEC>(sampleCount);
        // center of gaussian pulse is at base_t0
        double baseT0 = baseStart + this->w_ * 3.0;
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            (*wave)[i] = (REAL)exp((-pow(baseT - baseT0, 2.0)) / (2.0 * this->sigma_ * this->sigma_));
        }
        return wave;
    }
    COMPLEX_VEC_SP complexModel(double baseStart, size_t sampleCount, double resolution) override
    {
        COMPLEX_VEC_SP wave = std::make_shared<COMPLEX_VEC>(sampleCount);
        // center of gaussian pulse is at base_t0
        double baseT0 = baseStart + this->w_ * 3.0;
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            (*wave)[i] = (COMPLEX)(REAL)exp((-pow(baseT - baseT0, 2.0)) / (2.0 * this->sigma_ * this->sigma_));
        }
        return wave;
    }
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        *serializedData += std::string((char *)(&t0_), sizeof(double));
        *serializedData += std::string((char *)(&w_), sizeof(double));
        *serializedData += std::string((char *)(&amp_), sizeof(REAL));
        void *envPtr = nullptr;
        return std::make_tuple(serializedData, envPtr);
    }
    Gaussian(std::string &serialized, size_t &current)
        : Gaussian(*(double *)(&serialized[current]),
                   *(double *)(&serialized[current + sizeof(double)]),
                   *(REAL *)(&serialized[current + sizeof(double) * 2]))
    {
        current += sizeof(double) * 2 + sizeof(REAL);
    }
};

class GaussianDRAG : public Envelope
{
public:
    double t0_;
    double w_;
    double sigma_;
    double coef_;
    double df_;
    double phase_;

    GaussianDRAG(double t0, double w, REAL amp, double coef, double df, double phase)
        : Envelope(t0 - 3 * w, t0 + 3 * w, true, amp, "GaussianDRAG"),
          t0_(t0), w_(w), coef_(coef), df_(df), phase_(phase)
    {
        sigma_ = w_ / sqrt(8.0 * log(2.0));
        feature_.values = std::make_shared<VEC<LLINT>>(3);
        feature_.values->at(0) = (LLINT)ROUND(w_ / TIME_ATOL);
        feature_.values->at(1) = (LLINT)ROUND(coef_ / FLOAT_ATOL);
        feature_.values->at(2) = (LLINT)ROUND(df_ / FREQ_ATOL);
        type_ = EnvType::GAUSSIANDRAG;
    }
    COMPLEX_VEC_SP complexModel(double baseStart, size_t sampleCount, double resolution) override
    {
        COMPLEX_VEC_SP wave = std::make_shared<COMPLEX_VEC>(sampleCount);
        double baseT0 = baseStart + this->w_ * 3.0;
        double baseT;
        double phase;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            phase = -2 * M_PI * this->df_ * baseT;
            (*wave)[i] =
                COMPLEX((REAL)cos(phase), (REAL)sin(phase)) *
                (COMPLEX)(exp(-pow(baseT - baseT0, 2.0) / 2.0 / pow(this->sigma_, 2.0)) *
                          (1.0 - I * this->coef_ * (baseT - baseT0) / pow(this->sigma_, 2.0)));
        }
        return wave;
    }
    TP<double, SP<LazyArray<COMPLEX>>> complexTimeFunc(WaveCache &wc, double dt) override
    {
        auto [t, wave] = Envelope::complexTimeFunc(wc, dt);
        double phase = -2 * M_PI * this->df_ * t + this->phase_;
        *wave *= COMPLEX((REAL)cos(phase), (REAL)sin(phase));
        return std::make_tuple(t, wave);
    }
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        *serializedData += std::string((char *)(&t0_), sizeof(double));
        *serializedData += std::string((char *)(&w_), sizeof(double));
        *serializedData += std::string((char *)(&amp_), sizeof(REAL));
        *serializedData += std::string((char *)(&coef_), sizeof(double));
        *serializedData += std::string((char *)(&df_), sizeof(double));
        *serializedData += std::string((char *)(&phase_), sizeof(double));
        void *envPtr = nullptr;
        return std::make_tuple(serializedData, envPtr);
    }
    GaussianDRAG(std::string &serialized, size_t &current)
        : GaussianDRAG(*(double *)(&serialized[current]),
                       *(double *)(&serialized[current + sizeof(double)]),
                       *(REAL *)(&serialized[current + sizeof(double) * 2]),
                       *(double *)(&serialized[current + sizeof(double) * 2 + sizeof(REAL)]),
                       *(double *)(&serialized[current + sizeof(double) * 3 + sizeof(REAL)]),
                       *(double *)(&serialized[current + sizeof(double) * 4 + sizeof(REAL)]))
    {
        current += sizeof(double) * 5 + sizeof(REAL);
    }
};

class CosineDRAG : public Envelope
{
public:
    double t0_;
    double w_;
    double coef_;
    double df_;
    double phase_;

    CosineDRAG(double t0, double w, REAL amp, double coef, double df, double phase)
        : Envelope(t0 - w / 2, t0 + w / 2, true, amp, "CosineDRAG"),
          t0_(t0), w_(w), coef_(coef), df_(df), phase_(phase)
    {
        feature_.values = std::make_shared<VEC<LLINT>>(3);
        feature_.values->at(0) = (LLINT)ROUND(w_ / TIME_ATOL);
        feature_.values->at(1) = (LLINT)ROUND(coef_ / FLOAT_ATOL);
        feature_.values->at(2) = (LLINT)ROUND(df_ / FREQ_ATOL);
        type_ = EnvType::COSINEDRAG;
    }
    COMPLEX_VEC_SP complexModel(double baseStart, size_t sampleCount, double resolution) override
    {
        COMPLEX_VEC_SP wave = std::make_shared<COMPLEX_VEC>(sampleCount);
        double baseT0 = baseStart + this->w_ / 2.0;
        double baseT;
        double phase;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            phase = -2 * M_PI * this->df_ * baseT;
            (*wave)[i] =
                COMPLEX((REAL)cos(phase), (REAL)sin(phase)) *
                (COMPLEX)(0.5 * (1 + cos(2 * M_PI * (baseT - baseT0) / this->w_)) -
                          I * this->coef_ * M_PI / this->w_ *
                              sin(2 * M_PI * (baseT - baseT0) / this->w_)) *
                (COMPLEX)(baseT - baseT0 > -this->w_ / 2.0 && baseT - baseT0 < this->w_ / 2.0);
        }
        return wave;
    }
    TP<double, SP<LazyArray<COMPLEX>>> complexTimeFunc(WaveCache &wc, double dt) override
    {
        auto [t, wave] = Envelope::complexTimeFunc(wc, dt);
        double phase = -2 * M_PI * this->df_ * t + this->phase_;
        *wave *= COMPLEX((REAL)cos(phase), (REAL)sin(phase));
        return std::make_tuple(t, wave);
    }
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        *serializedData += std::string((char *)(&t0_), sizeof(double));
        *serializedData += std::string((char *)(&w_), sizeof(double));
        *serializedData += std::string((char *)(&amp_), sizeof(REAL));
        *serializedData += std::string((char *)(&coef_), sizeof(double));
        *serializedData += std::string((char *)(&df_), sizeof(double));
        *serializedData += std::string((char *)(&phase_), sizeof(double));
        void *envPtr = nullptr;
        return std::make_tuple(serializedData, envPtr);
    }
    CosineDRAG(std::string &serialized, size_t &current)
        : CosineDRAG(*(double *)(&serialized[current]),
                     *(double *)(&serialized[current + sizeof(double)]),
                     *(REAL *)(&serialized[current + sizeof(double) * 2]),
                     *(double *)(&serialized[current + sizeof(double) * 2 + sizeof(REAL)]),
                     *(double *)(&serialized[current + sizeof(double) * 3 + sizeof(REAL)]),
                     *(double *)(&serialized[current + sizeof(double) * 4 + sizeof(REAL)]))
    {
        current += sizeof(double) * 5 + sizeof(REAL);
    }
};

class Triangle : public Envelope
{
public:
    double t0_;
    double tlen_;
    bool fall_;

    Triangle(double t0, double tlen, REAL amp, bool fall)
        : Envelope(t0, t0 + tlen, false, amp, "Triangle"), t0_(t0), tlen_(tlen), fall_(fall)
    {
        feature_.values = std::make_shared<VEC<LLINT>>(2);
        feature_.values->at(0) = (LLINT)ROUND(tlen_ / TIME_ATOL);
        feature_.values->at(1) = (LLINT)fall_;
        type_ = EnvType::TRIANGLE;
    }
    REAL_VEC_SP realModel(double baseStart, size_t sampleCount, double resolution) override
    {
        REAL_VEC_SP wave = std::make_shared<REAL_VEC>(sampleCount);
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            if (this->fall_)
            {
                (*wave)[i] =
                    (REAL)((1.0 - (baseT - baseStart) / this->tlen_) * ((baseT - baseStart) >= 0.0 && (baseT - baseStart) < this->tlen_));
            }
            else
            {
                (*wave)[i] =
                    (REAL)((baseT - baseStart) / this->tlen_ * ((baseT - baseStart) >= 0.0 && (baseT - baseStart) < this->tlen_));
            }
        }
        return wave;
    }
    COMPLEX_VEC_SP complexModel(double baseStart, size_t sampleCount, double resolution) override
    {
        COMPLEX_VEC_SP wave = std::make_shared<COMPLEX_VEC>(sampleCount);
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            if (this->fall_)
            {
                (*wave)[i] =
                    (COMPLEX)(REAL)((1.0 - (baseT - baseStart) / this->tlen_) * ((baseT - baseStart) >= 0.0 && (baseT - baseStart) < this->tlen_));
            }
            else
            {
                (*wave)[i] =
                    (COMPLEX)(REAL)((baseT - baseStart) / this->tlen_ * ((baseT - baseStart) >= 0.0 && (baseT - baseStart) < this->tlen_));
            }
        }
        return wave;
    }
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        *serializedData += std::string((char *)(&t0_), sizeof(double));
        *serializedData += std::string((char *)(&tlen_), sizeof(double));
        *serializedData += std::string((char *)(&amp_), sizeof(REAL));
        *serializedData += std::string((char *)(&fall_), sizeof(bool));
        void *envPtr = nullptr;
        return std::make_tuple(serializedData, envPtr);
    }
    Triangle(std::string &serialized, size_t &current)
        : Triangle(*(double *)(&serialized[current]),
                   *(double *)(&serialized[current + sizeof(double)]),
                   *(REAL *)(&serialized[current + sizeof(double) * 2]),
                   *(bool *)(&serialized[current + sizeof(double) * 2 + sizeof(REAL)]))
    {
        current += sizeof(double) * 2 + sizeof(REAL) + sizeof(bool);
    }
};

class Rect : public Envelope
{
public:
    double t0_;
    double tlen_;

    Rect(double t0, double tlen, REAL amp)
        : Envelope(t0, t0 + tlen, false, amp, "Rect"), t0_(t0), tlen_(tlen)
    {
        feature_.values = std::make_shared<VEC<LLINT>>(1);
        feature_.values->at(0) = (LLINT)ROUND(tlen_ / TIME_ATOL);
        type_ = EnvType::RECT;
    }
    REAL_VEC_SP realModel(double baseStart, size_t sampleCount, double resolution) override
    {
        REAL_VEC_SP wave = std::make_shared<REAL_VEC>(sampleCount);
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            (*wave)[i] = (REAL)((baseT - baseStart) >= 0.0 && (baseT - baseStart) < this->tlen_);
        }
        return wave;
    }
    COMPLEX_VEC_SP complexModel(double baseStart, size_t sampleCount, double resolution) override
    {
        COMPLEX_VEC_SP wave = std::make_shared<COMPLEX_VEC>(sampleCount);
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            (*wave)[i] =
                (COMPLEX)((baseT - baseStart) >= 0.0 && (baseT - baseStart) < this->tlen_);
        }
        return wave;
    }
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        *serializedData += std::string((char *)(&t0_), sizeof(double));
        *serializedData += std::string((char *)(&tlen_), sizeof(double));
        *serializedData += std::string((char *)(&amp_), sizeof(REAL));
        void *envPtr = nullptr;
        return std::make_tuple(serializedData, envPtr);
    }
    Rect(std::string &serialized, size_t &current)
        : Rect(*(double *)(&serialized[current]),
               *(double *)(&serialized[current + sizeof(double)]),
               *(REAL *)(&serialized[current + sizeof(double) * 2]))
    {
        current += sizeof(double) * 2 + sizeof(REAL);
    }
};

class Flattop : public Envelope
{
public:
    double t0_;
    double tlen_;
    double wLeft_;
    double wRight_;
    double aLeft_;
    double aRight_;

    Flattop(double t0, double tlen, double wLeft, double wRight, REAL amp)
        : Envelope(t0 - 3 * wLeft, t0 + tlen + 3 * wRight, false, amp, "Flattop"),
          t0_(t0), tlen_(tlen), wLeft_(wLeft), wRight_(wRight)
    {
        aLeft_ = 2 * sqrt(log(2)) / wLeft_;
        aRight_ = 2 * sqrt(log(2)) / wRight_;
        feature_.values = std::make_shared<VEC<LLINT>>(3);
        feature_.values->at(0) = (LLINT)ROUND(wLeft_ / TIME_ATOL);
        feature_.values->at(1) = (LLINT)ROUND(wRight_ / TIME_ATOL);
        feature_.values->at(2) = (LLINT)ROUND(tlen_ / TIME_ATOL);
        type_ = EnvType::FLATTOP;
    }
    REAL_VEC_SP realModel(double baseStart, size_t sampleCount, double resolution) override
    {
        REAL_VEC_SP wave = std::make_shared<REAL_VEC>(sampleCount);
        double baseT0 = baseStart + 3 * wLeft_;
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            (*wave)[i] = (REAL)((erf(aRight_ * (baseT0 + tlen_ - baseT)) - erf(aLeft_ * (baseT0 - baseT))) / 2.0);
        }
        return wave;
    }
    COMPLEX_VEC_SP complexModel(double baseStart, size_t sampleCount, double resolution) override
    {
        COMPLEX_VEC_SP wave = std::make_shared<COMPLEX_VEC>(sampleCount);
        double baseT0 = baseStart + 3 * wLeft_;
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            (*wave)[i] = (COMPLEX)(REAL)((erf(aRight_ * (baseT0 + tlen_ - baseT)) - erf(aLeft_ * (baseT0 - baseT))) / 2.0);
        }
        return wave;
    }
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        *serializedData += std::string((char *)(&t0_), sizeof(double));
        *serializedData += std::string((char *)(&tlen_), sizeof(double));
        *serializedData += std::string((char *)(&wLeft_), sizeof(double));
        *serializedData += std::string((char *)(&wRight_), sizeof(double));
        *serializedData += std::string((char *)(&amp_), sizeof(REAL));
        void *envPtr = nullptr;
        return std::make_tuple(serializedData, envPtr);
    }
    Flattop(std::string &serialized, size_t &current)
        : Flattop(*(double *)(&serialized[current]),
                  *(double *)(&serialized[current + sizeof(double)]),
                  *(double *)(&serialized[current + sizeof(double) * 2]),
                  *(double *)(&serialized[current + sizeof(double) * 3]),
                  *(REAL *)(&serialized[current + sizeof(double) * 4]))
    {
        current += sizeof(double) * 4 + sizeof(REAL);
    }
};

class RippleRect : public Envelope
{
public:
    double t0_;
    double tlen_;
    double w_;
    double ripples0_;
    double ripples1_;
    double ripples2_;
    double ripples3_;

    RippleRect(double t0, double tlen, REAL amp, double w,
               double ripples0, double ripples1, double ripples2, double ripples3)
        : Envelope(t0 - 3 * w, t0 + tlen + 3 * w, false, amp, "RippleRect"),
          t0_(t0), tlen_(tlen), w_(w),
          ripples0_(ripples0), ripples1_(ripples1),
          ripples2_(ripples2), ripples3_(ripples3)
    {
        feature_.values = std::make_shared<VEC<LLINT>>(6);
        feature_.values->at(0) = (LLINT)ROUND(ripples0_ / TIME_ATOL);
        feature_.values->at(1) = (LLINT)ROUND(ripples1_ / TIME_ATOL);
        feature_.values->at(2) = (LLINT)ROUND(ripples2_ / TIME_ATOL);
        feature_.values->at(3) = (LLINT)ROUND(ripples3_ / TIME_ATOL);
        feature_.values->at(4) = (LLINT)ROUND(w_ / TIME_ATOL);
        feature_.values->at(5) = (LLINT)ROUND(tlen_ / TIME_ATOL);
        type_ = EnvType::RIPPLERECT;
    }
    REAL_VEC_SP realModel(double baseStart, size_t sampleCount, double resolution) override
    {
        REAL_VEC_SP wave = std::make_shared<REAL_VEC>(sampleCount);
        double baseTmin = baseStart + 3.0 * w_;
        double baseTmax = baseTmin + tlen_;
        double baseTmid = (baseTmin + baseTmax) / 2.0;
        double baseAmp = 1.0 - ripples1_ - ripples3_;
        double sigma = (w_ + 1e-10) / 2.0 / sqrt(log(2.0)); // NOT the sigma of Gaussian func
        VEC<double> ripples = {ripples0_, ripples1_, ripples2_, ripples3_};
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            (*wave)[i] = (REAL)(baseAmp / 2 * (erf((baseT - baseTmin) / sigma) - erf((baseT - baseTmax) / sigma)));
            for (size_t idx = 0; idx < 4; idx++)
            {
                double r = ripples[idx];
                double idx_r = pow(2, idx / 2);
                if (idx % 2 == 0)
                {
                    (*wave)[i] += (REAL)(r / 2.0 * exp(-pow((idx_r * M_PI * sigma / 2.0 / tlen_), 2.0)) *
                                         (exp(I * idx_r * M_PI * (baseT - baseTmid) / tlen_) *
                                          (cerf((baseT - baseTmin) / sigma + I * idx_r * M_PI * sigma / 2.0 / tlen_) -
                                           cerf((baseT - baseTmax) / sigma + I * idx_r * M_PI * sigma / 2.0 / tlen_)))
                                             .imag());
                }
                else
                {
                    (*wave)[i] += (REAL)(r / 2.0 * exp(-pow((idx_r * M_PI * sigma / 2.0 / tlen_), 2.0)) *
                                         (exp(I * idx_r * M_PI * (baseT - baseTmid) / tlen_) *
                                          (cerf((baseT - baseTmin) / sigma + I * idx_r * M_PI * sigma / 2.0 / tlen_) -
                                           cerf((baseT - baseTmax) / sigma + I * idx_r * M_PI * sigma / 2.0 / tlen_)))
                                             .real());
                }
            }
        }
        return wave;
    }
    COMPLEX_VEC_SP complexModel(double baseStart, size_t sampleCount, double resolution) override
    {
        COMPLEX_VEC_SP wave = std::make_shared<COMPLEX_VEC>(sampleCount);
        double baseTmin = baseStart + 3.0 * w_;
        double baseTmax = baseTmin + tlen_;
        double baseTmid = (baseTmin + baseTmax) / 2.0;
        double baseAmp = 1.0 - ripples1_ - ripples3_;
        double sigma = (w_ + 1e-10) / 2.0 / sqrt(log(2.0)); // NOT the sigma of Gaussian func
        VEC<double> ripples = {ripples0_, ripples1_, ripples2_, ripples3_};
        double baseT;
        for (size_t i = 0; i < sampleCount; i++)
        {
            baseT = i * resolution;
            (*wave)[i] = (COMPLEX)(REAL)(baseAmp / 2.0 * (erf((baseT - baseTmin) / sigma) - erf((baseT - baseTmax) / sigma)));
            for (size_t idx = 0; idx < 4; idx++)
            {
                double r = ripples[0];
                double idx_r = pow(2, idx / 2);
                if (idx % 2 == 0)
                {
                    (*wave)[i] += (COMPLEX)(REAL)(r / 2.0 * exp(-pow((idx_r * M_PI * sigma / 2.0 / tlen_), 2.0)) *
                                                  (exp(I * idx_r * M_PI * (baseT - baseTmid) / tlen_) *
                                                   (cerf((baseT - baseTmin) / sigma + I * idx_r * M_PI * sigma / 2.0 / tlen_) -
                                                    cerf((baseT - baseTmax) / sigma + I * idx_r * M_PI * sigma / 2.0 / tlen_)))
                                                      .imag());
                }
                else
                {
                    (*wave)[i] += (COMPLEX)(REAL)(r / 2.0 * exp(-pow((idx_r * M_PI * sigma / 2.0 / tlen_), 2.0)) *
                                                  (exp(I * idx_r * M_PI * (baseT - baseTmid) / tlen_) *
                                                   (cerf((baseT - baseTmin) / sigma + I * idx_r * M_PI * sigma / 2.0 / tlen_) -
                                                    cerf((baseT - baseTmax) / sigma + I * idx_r * M_PI * sigma / 2.0 / tlen_)))
                                                      .real());
                }
            }
        }
        return wave;
    }
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        *serializedData += std::string((char *)(&t0_), sizeof(double));
        *serializedData += std::string((char *)(&tlen_), sizeof(double));
        *serializedData += std::string((char *)(&amp_), sizeof(REAL));
        *serializedData += std::string((char *)(&w_), sizeof(double));
        *serializedData += std::string((char *)(&ripples0_), sizeof(double));
        *serializedData += std::string((char *)(&ripples1_), sizeof(double));
        *serializedData += std::string((char *)(&ripples2_), sizeof(double));
        *serializedData += std::string((char *)(&ripples3_), sizeof(double));
        void *envPtr = nullptr;
        return std::make_tuple(serializedData, envPtr);
    }
    RippleRect(std::string &serialized, size_t &current)
        : RippleRect(*(double *)(&serialized[current]),
                     *(double *)(&serialized[current + sizeof(double)]),
                     *(REAL *)(&serialized[current + sizeof(double) * 2]),
                     *(double *)(&serialized[current + sizeof(double) * 2 + sizeof(REAL)]),
                     *(double *)(&serialized[current + sizeof(double) * 3 + sizeof(REAL)]),
                     *(double *)(&serialized[current + sizeof(double) * 4 + sizeof(REAL)]),
                     *(double *)(&serialized[current + sizeof(double) * 5 + sizeof(REAL)]),
                     *(double *)(&serialized[current + sizeof(double) * 6 + sizeof(REAL)]))
    {
        current += sizeof(double) * 7 + sizeof(REAL);
    }
};

class MixExp : public Envelope
/*
    Design for mixing envelope, with frequency df and phase shift.
    DO NOT MODIFY THIS CLASS OR USE IT DIRECTLY.
*/
{
public:
    double df_;
    double phase_;
    bool dynamical_;

    MixExp(double start, double end, double df, double phase, bool dynamical)
        : Envelope(start, end, true, 1.0, "MixExp"),
          df_(df), phase_(phase), dynamical_(dynamical)
    {
        feature_.values = std::make_shared<VEC<LLINT>>(2);
        feature_.values->at(0) = (LLINT)ROUND(df_ / FREQ_ATOL);
        feature_.values->at(1) = (LLINT)ROUND(this->duration() / FLOAT_ATOL);
        type_ = EnvType::MIXEXP;
    }
    TP<double, SP<LazyArray<REAL>>> realTimeFunc(WaveCache &wc, double dt) override
    {
        throw std::runtime_error("COMPLEX " + this->toString() + " should not call realTimeFunc!");
    }
    TP<double, SP<LazyArray<COMPLEX>>> complexTimeFunc(WaveCache &wc, double dt) override
    {
        double shiftT = ROUND((this->start() + dt) / TIME_ATOL) / wc.resolutionOverAtol_ * wc.resolution_;
        auto cache = wc.getComplexCache(this->toString());
        EnvFeature feature = this->feature();
        if (cache->count(feature) == 0)
        {
            size_t sampleCount = ROUND(std::floor(this->duration() / wc.resolution_ + 0.5)) + END_PADDING;
            COMPLEX_VEC_SP wave = std::make_shared<COMPLEX_VEC>(sampleCount);
            double baseT;
            double phase;
            for (size_t i = 0; i < sampleCount; i++)
            {
                baseT = i * wc.resolution_;
                phase = -2 * M_PI * this->df_ * baseT;
                (*wave)[i] =
                    COMPLEX((REAL)cos(phase), (REAL)sin(phase));
            }
            (*cache)[feature] = wave;
        }
        if (this->dynamical_) // keep phase = 0 at t=0
        {
            double phase = -2.0 * M_PI * this->df_ * shiftT + this->phase_;
            return std::make_tuple(shiftT,
                                   std::make_shared<LazyArray<COMPLEX>>(COMPLEX((REAL)cos(phase), (REAL)sin(phase)), cache->at(feature)));
        }
        else // keep phase = 0 at t=dt
        {
            double phase = -2.0 * M_PI * this->df_ * (shiftT - dt) + this->phase_;
            return std::make_tuple(shiftT,
                                   std::make_shared<LazyArray<COMPLEX>>(COMPLEX((REAL)cos(phase), (REAL)sin(phase)), cache->at(feature)));
        }
    }
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        *serializedData += std::string((char *)(&start_), sizeof(double));
        *serializedData += std::string((char *)(&end_), sizeof(double));
        *serializedData += std::string((char *)(&df_), sizeof(double));
        *serializedData += std::string((char *)(&phase_), sizeof(double));
        *serializedData += std::string((char *)(&dynamical_), sizeof(bool));
        void *envPtr = nullptr;
        return std::make_tuple(serializedData, envPtr);
    }
    MixExp(std::string &serialized, size_t &current)
        : MixExp(*(double *)(&serialized[current]),
                 *(double *)(&serialized[current + sizeof(double)]),
                 *(double *)(&serialized[current + sizeof(double) * 2]),
                 *(double *)(&serialized[current + sizeof(double) * 3]),
                 *(bool *)(&serialized[current + sizeof(double) * 4]))
    {
        current += sizeof(double) * 4 + sizeof(bool);
    }
};

class EnvMix : public AbstractEnvelope
{
public:
    SP<AbstractEnvelope> env;
    SP<AbstractEnvelope> mixExp;

    EnvMix(SP<AbstractEnvelope> &env, double df, double phase, bool dynamical)
        : env(env), mixExp(std::make_shared<MixExp>(env->start(), env->end(), df, phase, dynamical)) { type_ = EnvType::ENVMIX; }
    double start() override { return env->start(); }
    double end() override { return env->end(); }
    bool is_complex() override { return true; }
    std::string toString() const override { return "EnvMix(" + this->env->toString() + ")"; }
    TP<double, SP<LazyArray<REAL>>> realTimeFunc(WaveCache &wc, double dt) override
    {
        throw std::runtime_error("COMPLEX " + this->toString() + " should not call realTimeFunc!");
    }
    TP<double, SP<LazyArray<COMPLEX>>> complexTimeFunc(WaveCache &wc, double dt) override
    {
        auto [t, wave] = env->complexTimeFunc(wc, dt);
        auto [_, mixWave] = mixExp->complexTimeFunc(wc, dt);
        *wave *= mixWave;
        return std::make_tuple(t, wave);
    }
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        VEC<void *> *envPtrs = new VEC<void *>(2);
        (*envPtrs)[0] = (void *)env.get();
        (*envPtrs)[1] = (void *)mixExp.get();
        return std::make_tuple(serializedData, (void *)envPtrs);
    }
    EnvMix(std::string &serialized, size_t &current) {}
    void setDeserializedEnvPtr(std::string &serialized, size_t &current, VEC<SP<AbstractEnvelope>> &DeserializedEnvs) override
    {
        SVIT index = *(SVIT *)(&serialized[current]);
        current += sizeof(SVIT);
        this->env = DeserializedEnvs[index];
        index = *(SVIT *)(&serialized[current]);
        current += sizeof(SVIT);
        this->mixExp = DeserializedEnvs[index];
    }
};

class EnvShift : public AbstractEnvelope
{
public:
    SP<AbstractEnvelope> env; // Should be basic envelopes like Gaussian, etc.
    double dt;

    EnvShift(SP<AbstractEnvelope> &env, double dt) : env(env), dt(dt) { type_ = EnvType::ENVSHIFT; }
    double start() override { return env->start() + dt; }
    double end() override { return env->end() + dt; }
    bool is_complex() override { return env->is_complex(); }
    std::string toString() const override { return "EnvShift(" + this->env->toString() + ")"; }
    SP<AbstractEnvelope> operator>>(const double &dt) override;
    TP<double, SP<LazyArray<REAL>>> realTimeFunc(WaveCache &wc, double dt) override;
    TP<double, SP<LazyArray<COMPLEX>>> complexTimeFunc(WaveCache &wc, double dt) override;
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        *serializedData += std::string((char *)(&dt), sizeof(double));
        void *envPtr = (void *)env.get();
        return std::make_tuple(serializedData, envPtr);
    }
    EnvShift(std::string &serialized, size_t &current) : dt(*(double *)(&serialized[current]))
    {
        current += sizeof(double);
    }
    void setDeserializedEnvPtr(std::string &serialized, size_t &current, VEC<SP<AbstractEnvelope>> &DeserializedEnvs) override
    {
        SVIT index = *(SVIT *)(&serialized[current]);
        current += sizeof(SVIT);
        this->env = DeserializedEnvs[index];
    }
};

template <typename T>
class EnvProd : public AbstractEnvelope
{
public:
    SP<AbstractEnvelope> env; // Should be basic envelopes like Gaussian, etc and assistant envelopes like EnvShift.
    T factor;

    EnvProd(SP<AbstractEnvelope> &env, T factor) : env(env), factor(factor) { type_ = EnvType::ENVPROD; }
    double start() override { return env->start(); }
    double end() override { return env->end(); }
    bool is_complex() override;
    std::string toString() const override { return "EnvProd(" + this->env->toString() + ")"; }
    SP<AbstractEnvelope> operator*(const REAL &scale) override;
    SP<AbstractEnvelope> operator*(const COMPLEX &scale) override;
    SP<AbstractEnvelope> operator>>(const double &dt) override;
    TP<double, SP<LazyArray<REAL>>> realTimeFunc(WaveCache &wc, double dt) override;
    TP<double, SP<LazyArray<COMPLEX>>> complexTimeFunc(WaveCache &wc, double dt) override;
    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        bool T_is_complex = std::is_same_v<T, COMPLEX>;
        *serializedData += std::string((char *)&T_is_complex, sizeof(bool));
        *serializedData += std::string((char *)(&factor), sizeof(T));
        void *envPtr = (void *)env.get();
        return std::make_tuple(serializedData, envPtr);
    }
    EnvProd(std::string &serialized, size_t &current) : factor(*(T *)(&serialized[current]))
    {
        current += sizeof(T);
    }
    void setDeserializedEnvPtr(std::string &serialized, size_t &current, VEC<SP<AbstractEnvelope>> &DeserializedEnvs) override
    {
        SVIT index = *(SVIT *)(&serialized[current]);
        current += sizeof(SVIT);
        this->env = DeserializedEnvs[index];
    }
};

class EnvSum : public AbstractEnvelope
{
public:
    VEC<SP<AbstractEnvelope>> envs; // Can be basic envelopes like Gaussian, etc and assistant envelopes like EnvShift, EnvProd, etc.

    EnvSum() { type_ = EnvType::ENVSUM; }
    double start() override;
    double end() override;
    bool is_complex() override;
    std::string toString() const override;
    SP<AbstractEnvelope> operator+(SP<AbstractEnvelope> &env) override;
    SP<AbstractEnvelope> operator+=(SP<AbstractEnvelope> &env) override;
    SP<AbstractEnvelope> operator*(const REAL &scale) override;
    SP<AbstractEnvelope> operator*(const COMPLEX &scale) override;
    SP<AbstractEnvelope> operator>>(const double &dt) override;
    TP<double, SP<LazyArray<REAL>>> realTimeFunc(WaveCache &wc, double dt) override;
    TP<double, SP<LazyArray<COMPLEX>>> complexTimeFunc(WaveCache &wc, double dt) override;

    TP<std::string *, void *> serializationState() const override
    {
        std::string *serializedData = new std::string((char *)&type_, sizeof(EnvType));
        size_t size_ = envs.size();
        *serializedData += std::string((char *)&size_, sizeof(size_t));
        VEC<void *> *envPtrs = new VEC<void *>(envs.size());
        for (size_t i = 0; i < size_; i++)
        {
            (*envPtrs)[i] = (void *)envs[i].get();
        }
        return std::make_tuple(serializedData, (void *)envPtrs);
    }
    EnvSum(std::string &serialized, size_t &current)
    {
        envs = VEC<SP<AbstractEnvelope>>(*(size_t *)&(serialized[current]));
        current += sizeof(size_t);
    }
    void setDeserializedEnvPtr(std::string &serialized, size_t &current, VEC<SP<AbstractEnvelope>> &DeserializedEnvs) override
    {
        for (auto &env : envs)
        {
            SVIT index = *(SVIT *)(&serialized[current]);
            current += sizeof(SVIT);
            env = DeserializedEnvs[index];
        }
    }
};

SP<AbstractEnvelope> mix(SP<AbstractEnvelope> &env, double df, double phase, bool dynamical)
{
    EnvType et = env->type_;
    SP<AbstractEnvelope> result;
    switch (et)
    {
    case EnvType::ENVSUM:
    {
        SP<EnvSum> envSum = std::dynamic_pointer_cast<EnvSum>(env);
        SP<EnvSum> newEnvSum = std::make_shared<EnvSum>();
        newEnvSum->envs.reserve(envSum->envs.size());
        for (auto &e : envSum->envs)
        {
            newEnvSum->envs.push_back(mix(e, df, phase, dynamical));
        }
        result = newEnvSum;
        break;
    }
    case EnvType::ENVPROD:
    {
        if (env->is_complex())
        {
            SP<EnvProd<COMPLEX>> envProd = std::dynamic_pointer_cast<EnvProd<COMPLEX>>(env);
            SP<EnvProd<COMPLEX>> newEnvProd =
                std::make_shared<EnvProd<COMPLEX>>(mix(envProd->env, df, phase, dynamical), envProd->factor);
            result = newEnvProd;
        }
        else
        {
            SP<EnvProd<REAL>> envProd = std::dynamic_pointer_cast<EnvProd<REAL>>(env);
            SP<EnvProd<REAL>> newEnvProd =
                std::make_shared<EnvProd<REAL>>(mix(envProd->env, df, phase, dynamical), envProd->factor);
            result = newEnvProd;
        }
        break;
    }
    default:
        result = std::make_shared<EnvMix>(env, df, phase, dynamical);
        break;
    }
    return result;
}

SP<AbstractEnvelope> align(SP<AbstractEnvelope> &env, VEC<double> &dt)
{
    EnvType et = env->type_;
    SP<EnvSum> result = std::make_shared<EnvSum>();
    switch (et)
    {
    case EnvType::ENVSUM:
    {
        SP<EnvSum> envSum = std::dynamic_pointer_cast<EnvSum>(env);
        for (auto &e : envSum->envs)
        {
            (*result) += align(e, dt);
        }
        break;
    }
    default:
        result->envs.reserve(dt.size());
        for (auto &dt_ : dt)
        {
            result->envs.push_back((*env) >> dt_);
        }
        break;
    }
    return result;
}

template <typename T>
SP<AbstractEnvelope> align(SP<AbstractEnvelope> &env, VEC<double> &dt, VEC<T> &amp)
{
    if (dt.size() != amp.size())
    {
        throw std::runtime_error("dt and amp should have the same size.");
    }
    EnvType et = env->type_;
    SP<EnvSum> result = std::make_shared<EnvSum>();
    switch (et)
    {
    case EnvType::ENVSUM:
    {
        SP<EnvSum> envSum = std::dynamic_pointer_cast<EnvSum>(env);
        for (auto &e : envSum->envs)
        {
            (*result) += align(e, dt, amp);
        }
        break;
    }
    default:
        result->envs.reserve(dt.size());
        for (size_t i = 0; i < dt.size(); i++)
        {
            result->envs.push_back((*((*env) >> dt[i])) * amp[i]);
        }
        break;
    }
    return result;
}

TP<double, ANY_VEC_SP, bool> decodeEnvelope(SP<AbstractEnvelope> &env, WaveCache &wc);

TP<double, ANY_VEC_SP, bool> decodeEnvelope(SP<AbstractEnvelope> &env, WaveCache &wc, double start, double end);

SP<std::string> serialization(SP<AbstractEnvelope> &env);

SP<AbstractEnvelope> deserialization(std::string &serialized);
