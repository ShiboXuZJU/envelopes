#include "envelopes.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ctime>

namespace py = pybind11;

template <typename T>
class VectorBuffer
{
public:
    SP<VEC<T>> vec_;
    VectorBuffer(SP<VEC<T>> vec) : vec_(vec) {}
};

TP<double, UNION<SP<VectorBuffer<REAL>>, SP<VectorBuffer<COMPLEX>>>>
PyDecodeEnvelope(SP<AbstractEnvelope> &env, WaveCache &wc)
{
    auto [waveStart, wave_any, is_complex] = decodeEnvelope(env, wc);
    if (is_complex)
    {
        auto wave = std::get<COMPLEX_VEC_SP>(wave_any);
        return std::make_tuple(waveStart, std::make_shared<VectorBuffer<COMPLEX>>(wave));
    }
    else
    {
        auto wave = std::get<REAL_VEC_SP>(wave_any);
        return std::make_tuple(waveStart, std::make_shared<VectorBuffer<REAL>>(wave));
    }
}

TP<double, UNION<SP<VectorBuffer<REAL>>, SP<VectorBuffer<COMPLEX>>>>
PyDecodeEnvelope(SP<AbstractEnvelope> &env, WaveCache &wc, double start, double end)
{
    auto [waveStart, wave_any, is_complex] = decodeEnvelope(env, wc, start, end);
    if (is_complex)
    {
        auto wave = std::get<COMPLEX_VEC_SP>(wave_any);
        return std::make_tuple(waveStart, std::make_shared<VectorBuffer<COMPLEX>>(wave));
    }
    else
    {
        auto wave = std::get<REAL_VEC_SP>(wave_any);
        return std::make_tuple(waveStart, std::make_shared<VectorBuffer<REAL>>(wave));
    }
}

py::bytes pySerialization(SP<AbstractEnvelope> &env)
{
    return (py::bytes)*serialization(env);
}

SP<AbstractEnvelope> pyDeserialization(py::bytes &serialized)
{
    return deserialization((std::string)serialized);
}

SP<AbstractEnvelope> pyAlign(SP<AbstractEnvelope> &env, py::array_t<double> &dt)
{
    std::vector<double> dt_vec(dt.data(), dt.data() + dt.shape(0));
    return align(env, dt_vec);
}

SP<AbstractEnvelope> pyAlign(SP<AbstractEnvelope> &env,
                             py::array_t<double> &dt,
                             py::array_t<REAL> &amp)
{
    std::vector<double> dt_vec(dt.data(), dt.data() + dt.shape(0));
    std::vector<REAL> amp_vec(amp.data(), amp.data() + amp.shape(0));
    return align<REAL>(env, dt_vec, amp_vec);
}

SP<AbstractEnvelope> pyAlign(SP<AbstractEnvelope> &env,
                             py::array_t<double> &dt,
                             py::array_t<COMPLEX> &amp)
{
    std::vector<double> dt_vec(dt.data(), dt.data() + dt.shape(0));
    std::vector<COMPLEX> amp_vec(amp.data(), amp.data() + amp.shape(0));
    return align<COMPLEX>(env, dt_vec, amp_vec);
}

VEC<SP<AbstractEnvelope>> pySplit(SP<AbstractEnvelope> &env,
                                  py::array_t<double> &starts,
                                  py::array_t<double> &ends)
{
    std::vector<double> starts_vec(starts.data(), starts.data() + starts.shape(0));
    std::vector<double> ends_vec(ends.data(), ends.data() + ends.shape(0));
    return *split(env, starts_vec, ends_vec);
}

PYBIND11_MODULE(envelopes_cpp, m)
{
    m.doc() = "Envelops impelemented in C++.";

    py::class_<VectorBuffer<REAL>, SP<VectorBuffer<REAL>>>(m, "RealVectorBuffer", py::buffer_protocol())
        .def_buffer([](VectorBuffer<REAL> &vec) -> py::buffer_info
                    { return py::buffer_info(
                          vec.vec_->data(),                      /* Pointer to buffer */
                          sizeof(REAL),                          /* Size of one scalar */
                          py::format_descriptor<REAL>::format(), /* Python struct-style format descriptor */
                          1,                                     /* Number of dimensions */
                          {vec.vec_->size()},                    /* Buffer dimensions */
                          {sizeof(REAL)}                         /* Strides (in bytes) for each index */
                      ); });
    py::class_<VectorBuffer<COMPLEX>, SP<VectorBuffer<COMPLEX>>>(m, "ComplexVectorBuffer", py::buffer_protocol())
        .def_buffer([](VectorBuffer<COMPLEX> &vec) -> py::buffer_info
                    { return py::buffer_info(
                          vec.vec_->data(),                         /* Pointer to buffer */
                          sizeof(COMPLEX),                          /* Size of one scalar */
                          py::format_descriptor<COMPLEX>::format(), /* Python struct-style format descriptor */
                          1,                                        /* Number of dimensions */
                          {vec.vec_->size()},                       /* Buffer dimensions */
                          {sizeof(COMPLEX)}                         /* Strides (in bytes) for each index */
                      ); });

    py::class_<WaveCache>(m, "WaveCache")
        .def(py::init<double>(), py::arg("resolution"))
        .def("resolution", &WaveCache::resolution, "Get the resolution of the cache.")
        .def("__repr__", &WaveCache::toString)
        .doc() = "Cache for waveforms. The same envelope might yield both real and complex wave data which will be cached separately.";

    py::class_<AbstractEnvelope, SP<AbstractEnvelope>>(m, "AbstractEnvelope")
        .def("start", &AbstractEnvelope::start)
        .def("end", &AbstractEnvelope::end)
        .def("is_complex", &AbstractEnvelope::is_complex)
        .def("__add__", py::overload_cast<const double &>(&AbstractEnvelope::operator+))
        .def("__add__", py::overload_cast<SP<AbstractEnvelope> &>(&AbstractEnvelope::operator+))
        .def("__radd__", py::overload_cast<const double &>(&AbstractEnvelope::operator+))
        .def("__radd__", py::overload_cast<SP<AbstractEnvelope> &>(&AbstractEnvelope::operator+))
        .def("__iadd__", &AbstractEnvelope::operator+=)
        .def("__mul__", py::overload_cast<const REAL &>(&AbstractEnvelope::operator*))
        .def("__mul__", py::overload_cast<const COMPLEX &>(&AbstractEnvelope::operator*))
        .def("__rmul__", py::overload_cast<const REAL &>(&AbstractEnvelope::operator*))
        .def("__rmul__", py::overload_cast<const COMPLEX &>(&AbstractEnvelope::operator*))
        .def("__rshift__", &AbstractEnvelope::operator>>)
        .def("__repr__", &AbstractEnvelope::toString);

    py::class_<Envelope, SP<Envelope>, AbstractEnvelope>(m, "Envelope");

    py::class_<Gaussian, SP<Gaussian>, Envelope>(m, "Gaussian")
        .def(py::init<double, double, REAL>(), py::arg("t0"), py::arg("w"), py::arg("amp"),
             "Parameters:\nt0 is the center of Gaussian, w is the width at half maximum, and amp is the amplitude.")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<Gaussian>(deserialization((std::string)s));
            }))
        .doc() = "Gaussian envelope.";

    py::class_<GaussianDRAG, SP<GaussianDRAG>, Envelope>(m, "GaussianDRAG")
        .def(py::init<double, double, REAL, double, double, double>(),
             py::arg("t0"), py::arg("w"), py::arg("amp"), py::arg("coef"), py::arg("df"), py::arg("phase"),
             "Parameters:\nt0 is the center of Gaussian, w is the width at half maximum, and amp is the amplitude.\nMathmatical formula: [Gaussin(t0, w, amp) + i * coef * GaussinDeriv(t0, w, amp)] * exp(i * (-2* pi * df * t + phase))")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<GaussianDRAG>(deserialization((std::string)s));
            }))
        .doc() = "Gaussian envelope shaped with DRAG.";

    py::class_<CosineDRAG, SP<CosineDRAG>, Envelope>(m, "CosineDRAG")
        .def(py::init<double, double, REAL, double, double, double>(),
             py::arg("t0"), py::arg("w"), py::arg("amp"), py::arg("coef"), py::arg("df"), py::arg("phase"),
             "Parameters:\nt0 is the center of Cosine, w is the width of Cosine, and amp is the amplitude.\nMathmatical formula: [Cosine(t0, w, amp) + i * coef * CosineDeriv(t0, w, amp)] * exp(i * (-2* pi * df * t + phase))")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<CosineDRAG>(deserialization((std::string)s));
            }))
        .doc() = "Cosine envelope shaped with DRAG.";

    py::class_<Triangle, SP<Triangle>, Envelope>(m, "Triangle")
        .def(py::init<double, double, REAL, bool>(),
             py::arg("t0"), py::arg("tlen"), py::arg("amp"), py::arg("fall"),
             "Parameters:\nt0 is the start of Triangle, tlen is the duriation, amp is the amplitude and fall indicates whether the Triangle is falling or rising.")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<Triangle>(deserialization((std::string)s));
            }))
        .doc() = "Triangle envelope.";

    py::class_<Rect, SP<Rect>, Envelope>(m, "Rect")
        .def(py::init<double, double, REAL>(),
             py::arg("t0"), py::arg("tlen"), py::arg("amp"),
             "Parameters:\nt0 is the start of Triangle, tlen is the duriation, and amp is the amplitude.")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<Rect>(deserialization((std::string)s));
            }))
        .doc() = "Rectangular envelope.";

    py::class_<Flattop, SP<Flattop>, Envelope>(m, "Flattop")
        .def(py::init<double, double, double, double, REAL>(),
             py::arg("t0"), py::arg("tlen"), py::arg("w_left"), py::arg("w_right"), py::arg("amp"),
             "Parameters:\nt0 is the start of Rect, tlen is the duriation, and amp is the amplitude.\nw_left(w_right) is the width of Gaussian kernel to convolve the left(right) edge of Rect.")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<Flattop>(deserialization((std::string)s));
            }))
        .doc() = "Rectangular envelope convolved with Gaussian kernel.";

    py::class_<RippleRect, SP<RippleRect>, Envelope>(m, "RippleRect")
        .def(py::init<double, double, REAL, double, double, double, double, double>(),
             py::arg("t0"), py::arg("tlen"), py::arg("amp"), py::arg("w"),
             py::arg("ripple0"), py::arg("ripple1"), py::arg("ripple2"), py::arg("ripple3"),
             "Parameters:\nt0 is the start of Rect, tlen is the duriation, and amp is the amplitude.\nripple0, ripple1, ripple2, ripple3 describe the ripples of the Rect.\nw is the width of Gaussian kernel to convolve the Rect.")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<RippleRect>(deserialization((std::string)s));
            }))
        .doc() = "Rectangular envelope with ripples convolved with Gaussian kernel.";

    py::class_<MixExp, SP<MixExp>, AbstractEnvelope>(m, "MixExp")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<MixExp>(deserialization((std::string)s));
            }));

    py::class_<EnvMix, SP<EnvMix>, AbstractEnvelope>(m, "EnvMix")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<EnvMix>(deserialization((std::string)s));
            }));

    py::class_<EnvShift, SP<EnvShift>, AbstractEnvelope>(m, "EnvShift")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<EnvShift>(deserialization((std::string)s));
            }));

    py::class_<EnvProd<REAL>, SP<EnvProd<REAL>>, AbstractEnvelope>(m, "RealEnvProd")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<EnvProd<REAL>>(deserialization((std::string)s));
            }));

    py::class_<EnvProd<COMPLEX>, SP<EnvProd<COMPLEX>>, AbstractEnvelope>(m, "COMPLEXEnvProd")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<EnvProd<COMPLEX>>(deserialization((std::string)s));
            }));

    py::class_<EnvSum, SP<EnvSum>, AbstractEnvelope>(m, "EnvSum")
        .def(py::pickle(
            [](SP<AbstractEnvelope> env)
            {
                return py::bytes(*serialization(env));
            },
            [](py::bytes &s)
            {
                return *std::dynamic_pointer_cast<EnvSum>(deserialization((std::string)s));
            }));
    m.def("decode_envelope", py::overload_cast<SP<AbstractEnvelope> &, WaveCache &>(&PyDecodeEnvelope),
          py::arg("env"), py::arg("wc"),
          "Decodes the envelope and returns the time and wave data.\nVectorBuffer is used to support numpy array access. (i.e. np.array(vec_buffer, copy=False).\nStart and end are extracted from env.start() and env.end().");
    m.def("decode_envelope", py::overload_cast<SP<AbstractEnvelope> &, WaveCache &, double, double>(&PyDecodeEnvelope),
          py::arg("env"), py::arg("wc"), py::arg("start"), py::arg("end"),
          "Decodes the envelope and returns the time and wave data.\nVectorBuffer is used to support numpy array access. (i.e. np.array(vec_buffer, copy=False)");
    m.def("serialization", &pySerialization, py::arg("env"),
          "Serializes the envelope to support pickle.dumps.");
    m.def("deserialization", &pyDeserialization, py::arg("serialized"),
          "Serializes the envelope to support pickle.loads.");
    m.def("align", py::overload_cast<SP<AbstractEnvelope> &, py::array_t<double> &>(&pyAlign),
          py::arg("env"), py::arg("dt"),
          "FAST aligning the envelope to the given time shifts.");
    m.def("align", py::overload_cast<SP<AbstractEnvelope> &, py::array_t<double> &, py::array_t<REAL> &>(&pyAlign),
          py::arg("env"), py::arg("dt"), py::arg("amp"),
          "FAST aligning the envelope to the given time shifts and real amplitudes.");
    m.def("align", py::overload_cast<SP<AbstractEnvelope> &, py::array_t<double> &, py::array_t<COMPLEX> &>(&pyAlign),
          py::arg("env"), py::arg("dt"), py::arg("amp"),
          "FAST aligning the envelope to the given time shifts and complex amplitudes.");
    m.def("split", &pySplit, py::arg("env"), py::arg("starts"), py::arg("ends"),
          "Split env according to starts and ends.\nThe provided starts and ends must be pre-sorted and have the same length.");
    m.def("mix", &mix, py::arg("env"), py::arg("df"), py::arg("phase"), py::arg("dynamical"),
          "FAST mixing the envelope by the given df and phase.\ndynamical=True means phase keeps 0 at t=0 (i.e. env * exp(i * (-2* pi * df * t + phase)).\ndynamical=False means phase keeps 0 at t=dt(env>>dt) (i.e. env * exp(i * (-2* pi * df * (t - dt) + phase)).");
}
