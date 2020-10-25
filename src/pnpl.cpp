#include "pnpl.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
namespace g2o{

typedef Eigen::Matrix<double,6,1,Eigen::ColMajor>   Vector6D;

class VertexSBALine : public BaseVertex<6, Vector6D>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    VertexSBALine();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
      _estimate.fill(0.);
    }

    virtual void oplusImpl(const double* update)
    {
      Eigen::Map<const Vector6D> v(update);
      _estimate += v;
    }
};

VertexSBALine::VertexSBALine() : BaseVertex<6, Vector6D>()
{
}

bool VertexSBALine::read(std::istream& is)
{
Vector6D lv;
for (int i=0; i<6; i++)
  is >> _estimate[i];
return true;
}

bool VertexSBALine::write(std::ostream& os) const
{
Vector6D lv=estimate();
for (int i=0; i<6; i++){
  os << lv[i] << " ";
}
return os.good();
}


class EdgeProjectLine : public  BaseBinaryEdge<4, Eigen::Vector4d, VertexSBALine, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectLine();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
      const VertexSBALine* v2 = static_cast<const VertexSBALine*>(_vertices[0]);
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
      Eigen::Vector4d obs(_measurement);
      Vector6D est = v2->estimate();
      Eigen::Vector2d u1 = cam->cam_map(v1->estimate().map(Eigen::Vector3d(est[0],est[1],est[2])));
      Eigen::Vector2d u2 = cam->cam_map(v1->estimate().map(Eigen::Vector3d(est[3],est[4],est[5])));
      double dx = obs[2] - obs[0];
      double dy = obs[3] - obs[1];
      double n = hypot(dx,dy);
      dx/=n;
      dy/=n;
      double d = -dy*obs[0]+dx*obs[1];
      double dist1 = -dy*u1[0]+dx*u1[1] - d;
      double dist2 = -dy*u2[0]+dx*u2[1] - d;
      _error = Eigen::Vector4d(dist1,dist2,0,0);
    }

    CameraParameters * _cam;
};

EdgeProjectLine::EdgeProjectLine() : BaseBinaryEdge<4, Eigen::Vector4d, VertexSBALine, VertexSE3Expmap>() {
  _cam = 0;
  resizeParameters(1);
  installParameter(_cam, 0);
}

bool EdgeProjectLine::read(std::istream& is){
  int paramId;
  is >> paramId;
  setParameterId(0, paramId);

  for (int i=0; i<4; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<4; i++)
    for (int j=i; j<4; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeProjectLine::write(std::ostream& os) const {
  os << _cam->id() << " ";
  for (int i=0; i<4; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<4; i++)
    for (int j=i; j<4; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
    return true;
}


} //namespace g2o


void PnPL(const std::vector<cv::Point3f>& pts3d, const std::vector<cv::Point2f>& pts2d,
          const std::vector<cv::Vec6f>& lns3d, const std::vector<cv::Vec4f>& lns2d,
          const cv::Mat& K,
          cv::Mat& R, cv::Mat& t)
{
    int npts = pts3d.size();
    int nlns = lns3d.size();
    bool DENSE = false;
    // init g2o
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (DENSE) {
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    } else {
    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    // optimizer.setVerbose(true);
    optimizer.setAlgorithm(solver);

    // add vertex for pose
    int vertex_id = 0;
    Eigen::Vector3d trans(0.1,0.1,0.1);
    Eigen::Quaterniond q;
    q.setIdentity();
    g2o::SE3Quat pose(q,trans);
    g2o::VertexSE3Expmap * v_se3 = new g2o::VertexSE3Expmap();
    v_se3->setId(vertex_id++);
    v_se3->setEstimate(pose);
    optimizer.addVertex(v_se3);

    // set camera intrinsic
    cv::Mat _K;
    K.convertTo(_K, CV_64FC1);
    g2o::CameraParameters * cam_params = new g2o::CameraParameters(_K.at<double>(0,0),
                                Eigen::Vector2d(_K.at<double>(0,2),_K.at<double>(1,2)), 0.);
    cam_params->setId(0);
    optimizer.addParameter(cam_params);

    // add vertex for points and edges for projection
    for(int i=0; i<npts; i++)
    {
        const auto& pt = pts3d[i];
        const auto& uv = pts2d[i];
        // printf("%.3f %.3f %.3f %.3f %.3f\n",pt.x, pt.y, pt.z, uv.x,uv.y);

        g2o::VertexSBAPointXYZ * v_p = new g2o::VertexSBAPointXYZ();
        v_p->setId(vertex_id++);
        v_p->setMarginalized(true);
        v_p->setEstimate(Eigen::Vector3d(pt.x, pt.y, pt.z));
        v_p->setFixed(true);
        optimizer.addVertex(v_p);

        g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
        e->setVertex(0, v_p);
        e->setVertex(1, v_se3);
        e->setMeasurement(Eigen::Vector2d(uv.x,uv.y));
        e->information() = Eigen::Matrix2d::Identity();
        e->setRobustKernel(new g2o::RobustKernelHuber);
        e->setParameterId(0, 0);
        optimizer.addEdge(e);
    }

    // add vertex for lines and edges for projection
    for(int i=0; i<nlns; i++)
    {
        const auto& ln3d = lns3d[i];
        const auto& ln2d = lns2d[i];
        // printf("%.3f %.3f %.3f %.3f %.3f\n",pt.x, pt.y, pt.z, uv.x,uv.y);

        g2o::VertexSBALine * v_l = new g2o::VertexSBALine();
        v_l->setId(vertex_id++);
        v_l->setMarginalized(true);
        g2o::Vector6D temp;
        temp<<ln3d[0],ln3d[1],ln3d[2],ln3d[3],ln3d[4],ln3d[5];
        v_l->setEstimate(temp);
        v_l->setFixed(true);
        optimizer.addVertex(v_l);

        g2o::EdgeProjectLine * e = new g2o::EdgeProjectLine();
        e->setVertex(0, v_l);
        e->setVertex(1, v_se3);
        e->setMeasurement(Eigen::Vector4d(ln2d[0],ln2d[1],ln2d[2],ln2d[3]));
        e->information() = Eigen::Matrix4d::Identity();
        e->setRobustKernel(new g2o::RobustKernelHuber);
        e->setParameterId(0, 0);
        optimizer.addEdge(e);
    }

    // optimize
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    // output result
    Eigen::MatrixXd T = Eigen::Isometry3d(v_se3->estimate()).matrix();
    R = (cv::Mat_<float>(3,3)<< T(0,0),T(0,1),T(0,2),
                                T(1,0),T(1,1),T(1,2),
                                T(2,0),T(2,1),T(2,2));
    t = (cv::Mat_<float>(3,1)<<T(0,3),T(1,3),T(2,3));
}

py::array_t<double> getPose(py::array_t<double> &pts3d_arr, py::array_t<double> &pts2d_arr,
          py::array_t<double> &lns3d_arr, py::array_t<double> &lns2d_arr,
          py::array_t<double> &K_arr)
{
    // int npts = pts3d.size();
    // int nlns = lns3d.size();
    
    py::buffer_info pts3d_buf = pts3d_arr.request();
    auto* pts3d_ptr = (double*)pts3d_buf.ptr;
    
    py::buffer_info pts2d_buf = pts2d_arr.request();
    auto* pts2d_ptr = (double*)pts2d_buf.ptr;
    
    py::buffer_info lns3d_buf = lns3d_arr.request();
    auto* lns3d_ptr = (double*)lns3d_buf.ptr;
    
    py::buffer_info lns2d_buf = lns2d_arr.request();
    auto* lns2d_ptr = (double*)lns2d_buf.ptr;
    
    py::buffer_info K_buf = K_arr.request();
    auto* K_ptr = (double*)K_buf.ptr;
    cv::Mat K = cv::Mat(K_buf.shape[0], K_buf.shape[1],  CV_32FC1, K_ptr);
    int npts = pts3d_buf.shape[0];
    int nlns = lns3d_buf.shape[0];
    std::cout << "points number: " << npts << std::endl;
    std::cout << "lines number: " << nlns << std::endl;
    /*for(int i = 0;i < npts;i++)
    {
        pts3d.push_back(cv::Point3f(pts3d_arr[0][0], pts3d_arr[0][1], pts3d_arr[0][2]));
        pts2d.push_back(cv::Point2f(pts2d_arr[0][0], pts2d_arr[0][1]));
    }
    for(int i = 0;i < nlns;i++)
    {
        lns3d.push_back(cv::Point6f(lns3d_arr[0][0], lns3d_arr[0][1], lns3d_arr[0][2], lns3d_arr[0][3], lns3d_arr[0][4], lns3d_arr[0][5]));
        lns2d.push_back(cv::Point4f(lns2d_arr[0][0], lns2d_arr[0][1], lns2d_arr[0][2], lns2d_arr[0][3]));
    }*/
    
    
    bool DENSE = false;
    // init g2o
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (DENSE) {
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    } else {
    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    // optimizer.setVerbose(true);
    optimizer.setAlgorithm(solver);

    // add vertex for pose
    int vertex_id = 0;
    Eigen::Vector3d trans(0.1,0.1,0.1);
    Eigen::Quaterniond q;
    q.setIdentity();
    g2o::SE3Quat pose(q,trans);
    g2o::VertexSE3Expmap * v_se3 = new g2o::VertexSE3Expmap();
    v_se3->setId(vertex_id++);
    v_se3->setEstimate(pose);
    optimizer.addVertex(v_se3);

    // set camera intrinsic
    cv::Mat _K;
    K.convertTo(_K, CV_64FC1);
    g2o::CameraParameters * cam_params = new g2o::CameraParameters(_K.at<double>(0,0),
                                Eigen::Vector2d(_K.at<double>(0,2),_K.at<double>(1,2)), 0.);
    cam_params->setId(0);
    optimizer.addParameter(cam_params);

    // add vertex for points and edges for projection
    for(int i=0; i<npts; i++)
    {
        const auto& pt = cv::Point3f(pts3d_ptr[i * 3 + 0], pts3d_ptr[i * 3 + 1], pts3d_ptr[i * 3 + 2]);
        const auto& uv = cv::Point2f(pts2d_ptr[i * 2 + 0], pts2d_ptr[i * 2 + 1]);
        // printf("%.3f %.3f %.3f %.3f %.3f\n",pt.x, pt.y, pt.z, uv.x,uv.y);

        g2o::VertexSBAPointXYZ * v_p = new g2o::VertexSBAPointXYZ();
        v_p->setId(vertex_id++);
        v_p->setMarginalized(true);
        v_p->setEstimate(Eigen::Vector3d(pt.x, pt.y, pt.z));
        v_p->setFixed(true);
        optimizer.addVertex(v_p);

        g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
        e->setVertex(0, v_p);
        e->setVertex(1, v_se3);
        e->setMeasurement(Eigen::Vector2d(uv.x,uv.y));
        e->information() = Eigen::Matrix2d::Identity();
        e->setRobustKernel(new g2o::RobustKernelHuber);
        e->setParameterId(0, 0);
        optimizer.addEdge(e);
    }

    // add vertex for lines and edges for projection
    for(int i=0; i<nlns; i++)
    {
        const auto& ln3d = cv::Vec6f(lns3d_ptr[i * 6 + 0], lns3d_ptr[i * 6 + 1], lns3d_ptr[i * 6 + 2], lns3d_ptr[i * 6 + 3], lns3d_ptr[i * 6 + 4], lns3d_ptr[i * 6 + 5]);
        const auto& ln2d = cv::Vec4f(lns2d_ptr[i * 4 + 0], lns2d_ptr[i * 4 + 1], lns2d_ptr[i * 4 + 2], lns2d_ptr[i * 4 + 3]);
        // printf("%.3f %.3f %.3f %.3f %.3f\n",pt.x, pt.y, pt.z, uv.x,uv.y);

        g2o::VertexSBALine * v_l = new g2o::VertexSBALine();
        v_l->setId(vertex_id++);
        v_l->setMarginalized(true);
        g2o::Vector6D temp;
        temp<<ln3d[0],ln3d[1],ln3d[2],ln3d[3],ln3d[4],ln3d[5];
        v_l->setEstimate(temp);
        v_l->setFixed(true);
        optimizer.addVertex(v_l);

        g2o::EdgeProjectLine * e = new g2o::EdgeProjectLine();
        e->setVertex(0, v_l);
        e->setVertex(1, v_se3);
        e->setMeasurement(Eigen::Vector4d(ln2d[0],ln2d[1],ln2d[2],ln2d[3]));
        e->information() = Eigen::Matrix4d::Identity();
        e->setRobustKernel(new g2o::RobustKernelHuber);
        e->setParameterId(0, 0);
        optimizer.addEdge(e);
    }

    // optimize
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    // output result
    Eigen::MatrixXd T = Eigen::Isometry3d(v_se3->estimate()).matrix();
    cv::Mat R = (cv::Mat_<double>(3,3)<< T(0,0),T(0,1),T(0,2),
                                T(1,0),T(1,1),T(1,2),
                                T(2,0),T(2,1),T(2,2));
    cv::Mat t = (cv::Mat_<double>(3,1)<<T(0,3),T(1,3),T(2,3));
    
    auto result = py::array_t<double>(12);
    result.resize({3, 4});
    py::buffer_info buf_result = result.request();
    double* ptr_result = (double*)buf_result.ptr;
    
    for(int i = 0;i < 3; i++)
    {
        for(int j = 0; j < 3;j++)
        {
            ptr_result[i * 4 + j] = T(i, j);
        }
        ptr_result[i * 4 + 3] = T(i, 3);
    }
    return result;
        
    
}

PYBIND11_MODULE(pnpl, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("getPose", &getPose, "A function to recover pose from pnp/pnl");
}
