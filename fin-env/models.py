from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base 

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    portfolio_items = relationship("PortfolioItem", back_populates="owner")


class PortfolioItem(Base):
    __tablename__ = 'portfolio_items'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    stock_ticker = Column(String, index=True)
    quantity = Column(Integer)
    purchase_price = Column(Float)

    owner = relationship("User", back_populates="portfolio_items")



